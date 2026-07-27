"""
Model Visualization — T7 Refresh for stamp 2026_05.

Generates:
  1. Top-20 Feature Importance bar chart (from RF base learner inside the voter)
  2. Confusion matrix heatmap (34×34, normalized rows)
  3. Per-class accuracy bar chart (Groundnut flagged if support = 0)
  4. Reliability diagram — raw vs calibrated (side-by-side)
  5. Confidence histogram — raw vs calibrated max-confidence distribution
  6. Season-filtered confusion matrices (Kharif / Rabi / Zaid / Annual)

All figures are stamped with the auto-discovered model stamp (e.g. 2026_05).

Usage:
    cd Crop_Recommendation_Engine
    python models/visualization.py
"""
import json
import sys
import warnings
from pathlib import Path

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
REGISTRY_DIR = BASE_DIR / "models" / "model_registry"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Central feature engineering (single source of truth) ───
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features
from generators.crop_profiles import CROP_TO_SEASON


# ─── Helpers ────────────────────────────────────────────────

def _write(f, msg=""):
    """Write a line to the report file and flush immediately."""
    f.write(msg + "\n")
    f.flush()


def _discover_stamp() -> str:
    """Auto-discover the latest model stamp from model_registry."""
    files = sorted(REGISTRY_DIR.glob("best_model_*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No best_model_*.pkl found in {REGISTRY_DIR}. "
            "Run baseline_models.py first."
        )
    return files[-1].stem.replace("best_model_", "")


def _load_calibrator(stamp: str):
    """Load calibrator dict and return (temperature, iso_calibrator, method)."""
    cal_data = joblib.load(REGISTRY_DIR / f"calibrator_{stamp}.pkl")
    method = cal_data.get("method", "temperature")
    temp = float(cal_data.get("temperature", 1.0))
    iso = cal_data.get("estimator", None)
    return method, temp, iso


def _apply_calibration(model, X, method, temp, iso):
    """Return (raw_probs, cal_probs) arrays."""
    raw_probs = model.predict_proba(X)
    if method == "isotonic" and iso is not None:
        cal_probs = iso.predict_proba(X)
    else:
        logits = np.log(np.clip(raw_probs, 1e-10, 1.0))
        scaled = logits / max(temp, 1e-6)
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        cal_probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    return raw_probs, cal_probs


def _extract_rf_importances(model, feat_cols):
    """Extract feature importances from inside the voting ensemble or plain RF."""
    # VotingClassifier wraps estimators in .estimators_
    # Try to find the RF base learner by name
    importances = None
    if hasattr(model, "estimators_"):
        # VotingClassifier: model.estimators_[i] is (name, fitted_est)
        for named in model.estimators_:
            est = named[1] if isinstance(named, tuple) else named
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                break
    if importances is None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    return importances


# ─── Main ───────────────────────────────────────────────────

def main():
    stamp = _discover_stamp()
    print(f"Auto-discovered model stamp: {stamp}")

    outpath = OUTPUT_DIR / "visualization_report.txt"

    with open(outpath, "w", encoding="utf-8") as f:
        try:
            _write(f, "=" * 70)
            _write(f, f"MODEL VISUALIZATION REPORT  (stamp={stamp})")
            _write(f, "=" * 70)

            # ── Load data ──
            df = pd.read_csv(FEATURES_CSV)
            class_counts = df["crop_label"].value_counts()
            rare = class_counts[class_counts < 5].index.tolist()
            if rare:
                _write(f, f"  Dropping rare classes (<5 samples): {rare}")
                df = df[~df["crop_label"].isin(rare)].reset_index(drop=True)
                le = LabelEncoder()
                df[TARGET_COL] = le.fit_transform(df["crop_label"])

            df = add_interaction_features(df)
            feat_cols = [c for c in df.columns
                         if c not in META_COLS and c != TARGET_COL]

            test_mask = df["season_year"] == 2025
            X_test = df.loc[test_mask, feat_cols].values
            y_test = df.loc[test_mask, TARGET_COL].values
            labels = sorted(df["crop_label"].unique())
            n_classes = len(labels)

            _write(f, f"\n  Test rows:  {test_mask.sum()}")
            _write(f, f"  Classes:    {n_classes} ({labels[:5]}…)")

            # ── Load model + calibrator ──
            model = joblib.load(REGISTRY_DIR / f"best_model_{stamp}.pkl")
            method, temp, iso = _load_calibrator(stamp)
            _write(f, f"\n  Calibration method: {method}  temperature: {temp:.3f}")

            raw_probs, cal_probs = _apply_calibration(model, X_test, method, temp, iso)
            y_pred = cal_probs.argmax(axis=1)

            # ═══════════════════════════════════════════════
            # 1. Feature Importance
            # ═══════════════════════════════════════════════
            _write(f, "\n[1] Feature Importance (Top 20 — RF base learner)")
            importances = _extract_rf_importances(model, feat_cols)

            if importances is not None and len(importances) == len(feat_cols):
                idx_sorted = np.argsort(importances)[::-1][:20]
                top_names = [feat_cols[i] for i in idx_sorted]
                top_vals = importances[idx_sorted]

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(range(len(top_names)), top_vals[::-1], color="#2e7d32")
                ax.set_yticks(range(len(top_names)))
                ax.set_yticklabels(top_names[::-1], fontsize=9)
                ax.set_xlabel("Importance (Gini)")
                ax.set_title(f"Top 20 Feature Importances — RF inside Voting Ensemble  (stamp={stamp})")
                plt.tight_layout()
                fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
                plt.close(fig)
                _write(f, "    Saved: evaluation/feature_importance.png")

                for i, fi in enumerate(idx_sorted[:10]):
                    _write(f, f"    {i+1:2d}. {feat_cols[fi]:35s} {importances[fi]:.4f}")
            else:
                _write(f, "    WARNING: feature_importances_ unavailable for this model type.")
                _write(f, f"    Model type: {type(model).__name__}")

            # ═══════════════════════════════════════════════
            # 2. Confusion Matrix (34×34)
            # ═══════════════════════════════════════════════
            _write(f, "\n[2] Confusion Matrix (normalized rows, 34-class)")
            cm = confusion_matrix(y_test, y_pred)
            # Guard for classes absent in test split (zero-row division)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1          # avoid division by zero
            cm_norm = cm.astype(float) / row_sums

            fig, ax = plt.subplots(figsize=(18, 16))
            sns.heatmap(cm_norm, annot=(n_classes <= 20), fmt=".2f",
                        cmap="YlGnBu",
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        linewidths=0.3, vmin=0, vmax=1,
                        annot_kws={"size": 6} if n_classes <= 20 else {})
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title(
                f"Normalized Confusion Matrix — Voting Ensemble  (stamp={stamp})\n"
                f"Test acc: {accuracy_score(y_test, y_pred)*100:.2f}%   "
                f"Macro-F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}",
                fontsize=13,
            )
            plt.xticks(rotation=45, ha="right", fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
            plt.close(fig)
            _write(f, "    Saved: evaluation/confusion_matrix.png")

            # ═══════════════════════════════════════════════
            # 3. Per-class Accuracy
            # ═══════════════════════════════════════════════
            _write(f, "\n[3] Per-class Accuracy")
            per_class_acc = cm_norm.diagonal()

            fig, ax = plt.subplots(figsize=(14, 5))
            colors = ["#e53935" if a < 0.85 else "#fb8c00" if a < 0.95 else "#43a047"
                      for a in per_class_acc]
            bars = ax.bar(range(n_classes), per_class_acc, color=colors)

            # Flag Groundnut if test support is zero
            for li, lab in enumerate(labels):
                if lab == "Groundnut" and cm.sum(axis=1)[li] == 0:
                    ax.text(li, 0.02, "⚠\nNo test\nsamples", ha="center",
                            va="bottom", fontsize=6, color="#e53935")

            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
            ax.set_ylabel("Per-class Accuracy")
            ax.set_title(f"Per-class Accuracy — Voting Ensemble  (stamp={stamp})")
            ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, linewidth=1, label="85% threshold")
            ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, linewidth=1, label="95% threshold")
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=150)
            plt.close(fig)
            _write(f, "    Saved: evaluation/per_class_accuracy.png")

            for lab, acc, sup in zip(labels, per_class_acc, cm.sum(axis=1)):
                flag = ("NO_TEST" if sup == 0 else
                        "LOW" if acc < 0.85 else
                        "OK" if acc < 0.95 else "HIGH")
                _write(f, f"    {lab:25s} {acc*100:5.1f}%  support={int(sup):4d}  [{flag}]")

            overall_acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
            _write(f, f"\n    Overall Accuracy: {overall_acc:.4f}")
            _write(f, f"    F1-macro:        {f1m:.4f}")

            # ═══════════════════════════════════════════════
            # 4. Reliability Diagram (raw vs calibrated)
            # ═══════════════════════════════════════════════
            _write(f, "\n[4] Reliability Diagram (Calibration)")
            try:
                max_raw = raw_probs.max(axis=1)
                max_cal = cal_probs.max(axis=1)
                correct = (y_pred == y_test).astype(float)

                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                ece_values = {}
                for ax, probs, title_label in [
                    (axes[0], max_raw, "Before Calibration (Raw)"),
                    (axes[1], max_cal, f"After Calibration ({method.capitalize()})"),
                ]:
                    bin_accs, bin_confs, bin_counts = [], [], []
                    for i in range(n_bins):
                        lo, hi = bin_edges[i], bin_edges[i + 1]
                        mask = ((probs >= lo) & (probs < hi)
                                if i < n_bins - 1
                                else (probs >= lo) & (probs <= hi))
                        if mask.sum() == 0:
                            bin_accs.append(0)
                            bin_confs.append((lo + hi) / 2)
                            bin_counts.append(0)
                        else:
                            bin_accs.append(correct[mask].mean())
                            bin_confs.append(probs[mask].mean())
                            bin_counts.append(mask.sum())

                    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6,
                           color="steelblue", edgecolor="navy", label="Accuracy in bin")
                    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect calibration")
                    ax.scatter(bin_confs, bin_accs, color="navy", s=30, zorder=5)
                    ax.set_xlabel("Mean Predicted Confidence")
                    ax.set_ylabel("Fraction Correct")
                    ax.set_title(title_label)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.05)
                    ax.legend(loc="lower right", fontsize=8)

                    total = sum(bin_counts)
                    ece = (sum(abs(a - c) * n / total
                               for a, c, n in zip(bin_accs, bin_confs, bin_counts)
                               if total > 0) if total > 0 else 0)
                    ece_values[title_label] = ece
                    ax.text(0.05, 0.92, f"ECE = {ece:.4f}",
                            transform=ax.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

                plt.suptitle(f"Reliability Diagram — Voting Ensemble  (stamp={stamp})", fontsize=14)
                plt.tight_layout()
                fig.savefig(OUTPUT_DIR / "reliability_diagram.png", dpi=150)
                plt.close(fig)
                _write(f, "    Saved: evaluation/reliability_diagram.png")
                for k, v in ece_values.items():
                    _write(f, f"    ECE ({k}): {v:.4f}")
            except Exception as cal_err:
                _write(f, f"    Skipped reliability diagram: {cal_err}")

            # ═══════════════════════════════════════════════
            # 5. Confidence Histogram (raw vs calibrated)      [NEW — T7]
            # ═══════════════════════════════════════════════
            _write(f, "\n[5] Confidence Histogram (raw vs calibrated)")
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                bins = np.linspace(0, 1, 26)

                for ax, probs, title_label, color in [
                    (axes[0], raw_probs.max(axis=1), "Raw Max Confidence", "#1565c0"),
                    (axes[1], cal_probs.max(axis=1), f"Calibrated Max Confidence ({method})", "#2e7d32"),
                ]:
                    ax.hist(probs, bins=bins, color=color, alpha=0.75, edgecolor="white")
                    mean_conf = probs.mean()
                    ax.axvline(mean_conf, color="red", linewidth=1.5,
                               linestyle="--", label=f"Mean = {mean_conf:.3f}")
                    ax.set_xlabel("Max Confidence")
                    ax.set_ylabel("Count")
                    ax.set_title(title_label)
                    ax.legend(fontsize=9)
                    ax.set_xlim(0, 1)

                plt.suptitle(f"Confidence Distribution — Voting Ensemble  (stamp={stamp})", fontsize=13)
                plt.tight_layout()
                fig.savefig(OUTPUT_DIR / "confidence_histogram.png", dpi=150)
                plt.close(fig)
                _write(f, "    Saved: evaluation/confidence_histogram.png")
                _write(f, f"    Raw  mean confidence: {raw_probs.max(axis=1).mean():.4f}")
                _write(f, f"    Cal. mean confidence: {cal_probs.max(axis=1).mean():.4f}")
            except Exception as hist_err:
                _write(f, f"    Skipped confidence histogram: {hist_err}")

            # ═══════════════════════════════════════════════
            # 6. Season-filtered Confusion Matrices           [NEW — T7]
            # ═══════════════════════════════════════════════
            _write(f, "\n[6] Season-filtered Confusion Matrices")
            try:
                # We need the raw crop label column from test rows
                test_df = df[test_mask].copy()
                # Map crop_label → season using CROP_TO_SEASON
                test_df["crop_season"] = test_df["crop_label"].map(
                    lambda c: CROP_TO_SEASON.get(c, "Unknown")
                )
                test_df["y_pred"] = y_pred
                test_df["y_true"] = y_test

                seasons = ["Kharif", "Rabi", "Zaid", "Annual"]
                fig, axes_grid = plt.subplots(2, 2, figsize=(22, 20))
                axes_flat = axes_grid.flatten()

                for ax, season in zip(axes_flat, seasons):
                    mask_s = test_df["crop_season"] == season
                    n_season = mask_s.sum()
                    if n_season == 0:
                        ax.text(0.5, 0.5, f"No test rows for {season}",
                                ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{season} (0 rows)")
                        continue

                    # Get the crops valid in this season
                    season_labels = sorted(test_df.loc[mask_s, "crop_label"].unique())
                    # Re-index predictions to crop names for this season's subset
                    y_t_s = test_df.loc[mask_s, "y_true"].values
                    y_p_s = test_df.loc[mask_s, "y_pred"].values

                    # Map encoded ints back to crop names
                    enc_to_name = {v: k for k, v in
                                   {lab: i for i, lab in enumerate(labels)}.items()}
                    y_t_names = [labels[v] if v < len(labels) else str(v) for v in y_t_s]
                    y_p_names = [labels[v] if v < len(labels) else str(v) for v in y_p_s]

                    cm_s = confusion_matrix(y_t_names, y_p_names, labels=season_labels)
                    row_s = cm_s.sum(axis=1, keepdims=True)
                    row_s[row_s == 0] = 1
                    cm_s_norm = cm_s.astype(float) / row_s

                    n_labels = len(season_labels)
                    annot = n_labels <= 15
                    sns.heatmap(cm_s_norm, annot=annot, fmt=".2f" if annot else "",
                                cmap="YlGnBu", xticklabels=season_labels,
                                yticklabels=season_labels, ax=ax,
                                linewidths=0.4, vmin=0, vmax=1,
                                annot_kws={"size": 7})
                    acc_s = accuracy_score(y_t_names, y_p_names)
                    f1_s = f1_score(y_t_names, y_p_names,
                                    average="macro", zero_division=0,
                                    labels=season_labels)
                    ax.set_title(
                        f"{season}  (n={n_season}, acc={acc_s*100:.1f}%, F1={f1_s:.3f})",
                        fontsize=10,
                    )
                    ax.set_xlabel("Predicted", fontsize=8)
                    ax.set_ylabel("Actual", fontsize=8)
                    ax.tick_params(axis="x", rotation=45, labelsize=7)
                    ax.tick_params(axis="y", rotation=0, labelsize=7)

                    _write(f, f"    {season:8s}: n={n_season}  acc={acc_s*100:.1f}%  F1={f1_s:.3f}")

                plt.suptitle(
                    f"Season-Filtered Confusion Matrices — Voting Ensemble  (stamp={stamp})",
                    fontsize=14,
                )
                plt.tight_layout()
                fig.savefig(OUTPUT_DIR / "season_confusion_matrix.png", dpi=150)
                plt.close(fig)
                _write(f, "    Saved: evaluation/season_confusion_matrix.png")
            except Exception as season_err:
                import traceback
                _write(f, f"    Skipped season confusion matrix: {season_err}")
                _write(f, traceback.format_exc())

            _write(f, f"\n{'=' * 70}")
            _write(f, "VISUALIZATION COMPLETE")
            _write(f, f"{'=' * 70}")

        except Exception as e:
            import traceback
            _write(f, f"\nERROR: {e}")
            traceback.print_exc(file=f)

    print(f"Results written to {outpath}")


if __name__ == "__main__":
    main()
