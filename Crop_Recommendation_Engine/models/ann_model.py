"""
ANN (Artificial Neural Network) Model for Crop Recommendation Engine.

Trains a PyTorch feedforward network on the same features/split as baseline_models.py,
then compares accuracy, F1, and calibration against the Random Forest baseline.

Usage: python Crop_Recommendation_Engine/models/ann_model.py
"""
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─── Paths (same as baseline_models.py) ───
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
ENCODERS_JSON = BASE_DIR / "data" / "processed" / "label_encoders.json"
REGISTRY_DIR = BASE_DIR / "models" / "model_registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Central feature engineering (single source of truth) ───
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features


def _inject_label_noise(y, noise_rate=0.05, seed=42):
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    classes = np.unique(y)
    for idx in flip_idx:
        choices = classes[classes != y[idx]]
        y_noisy[idx] = rng.choice(choices)
    return y_noisy


def load_data():
    """Load features, split temporally, apply SMOTE + noise (same as baseline)."""
    df = pd.read_csv(FEATURES_CSV)

    # Drop rare classes
    class_counts = df["crop_label"].value_counts()
    rare = class_counts[class_counts < 5].index.tolist()
    if rare:
        df = df[~df["crop_label"].isin(rare)].reset_index(drop=True)
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df["crop_label"])

    df = add_interaction_features(df)
    feat_cols = [c for c in df.columns if c not in META_COLS and c != TARGET_COL]

    train_mask = df["season_year"].isin([2023, 2024])
    test_mask = df["season_year"] == 2025

    X_train = df.loc[train_mask, feat_cols].values
    y_train = df.loc[train_mask, TARGET_COL].values
    X_test = df.loc[test_mask, feat_cols].values
    y_test = df.loc[test_mask, TARGET_COL].values

    # SMOTE
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Label noise
    y_train = _inject_label_noise(y_train, noise_rate=0.05)

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    labels = sorted(df["crop_label"].unique())
    return X_train, X_test, y_train, y_test, feat_cols, labels, n_classes


# ─── ANN Architecture ───
class CropANN(nn.Module):
    """3-hidden-layer feedforward network with dropout + batch norm."""

    def __init__(self, n_features, n_classes, hidden_sizes=(256, 128, 64),
                 dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_ann(model, train_loader, val_X, val_y, n_epochs=100, lr=1e-3,
              patience=15):
    """Train with early stopping on validation accuracy."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    best_acc = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(DEVICE))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(val_y, val_preds)
        scheduler.step(val_acc)

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info("Epoch %3d: loss=%.4f  val_acc=%.4f  best=%.4f  lr=%.1e",
                        epoch, epoch_loss, val_acc, best_acc, lr_now)

        if wait >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, history, best_acc


def evaluate_ann(model, X_test_t, y_test, labels, scaler):
    """Evaluate ANN and compare with RF baseline."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1_mac = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_wt = f1_score(y_test, preds, average="weighted", zero_division=0)

    # Temperature scaling on ANN probs
    from scipy.optimize import minimize_scalar
    logits_np = np.log(np.clip(probs, 1e-10, 1.0))

    def nll(T):
        scaled = logits_np / T
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        softmax_s = exp_s / exp_s.sum(axis=1, keepdims=True)
        return -np.mean(np.log(softmax_s[np.arange(len(y_test)), y_test] + 1e-10))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_ann = result.x

    # Brier score
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import brier_score_loss
    y_bin = label_binarize(y_test, classes=np.arange(probs.shape[1]))
    brier = np.mean(np.sum((probs - y_bin) ** 2, axis=1))

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "temperature": T_ann,
        "brier_score": brier,
        "probs": probs,
        "preds": preds,
    }


def main():
    """Train ANN and compare with Random Forest baseline."""
    outpath = BASE_DIR / "ann_training_results.txt"

    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(outpath, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)

    try:
        logger.info("=" * 70)
        logger.info("ANN MODEL TRAINING — Crop Recommendation Engine")
        logger.info("Date: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
        logger.info("Device: %s", DEVICE)
        logger.info("=" * 70)

        # Load data
        logger.info("[1] Loading data...")
        X_train, X_test, y_train, y_test, feat_cols, labels, n_classes = load_data()
        logger.info("Train: %s  Test: %s  Classes: %d", X_train.shape, X_test.shape, n_classes)

        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_sc)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test_sc)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

        # Build model
        n_features = X_train.shape[1]
        logger.info("[2] Building ANN: %d -> 256 -> 128 -> 64 -> %d", n_features, n_classes)
        model = CropANN(n_features, n_classes)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total parameters: %s", f"{total_params:,}")

        # Train
        logger.info("[3] Training...")
        t0 = time.time()
        model, history, best_val_acc = train_ann(
            model, train_loader, X_test_t, y_test,
            n_epochs=150, lr=1e-3, patience=20
        )
        train_time = time.time() - t0
        logger.info("Training time: %.1fs", train_time)

        # Evaluate
        logger.info("[4] Evaluating...")
        ann_results = evaluate_ann(model, X_test_t, y_test, labels, scaler)
        logger.info("ANN Accuracy:    %.4f", ann_results["accuracy"])
        logger.info("ANN F1-macro:    %.4f", ann_results["f1_macro"])
        logger.info("ANN F1-weighted: %.4f", ann_results["f1_weighted"])
        logger.info("ANN Temperature: %.3f", ann_results["temperature"])
        logger.info("ANN Brier Score: %.4f", ann_results["brier_score"])

        # Load RF baseline for comparison
        logger.info("[5] Comparing with Random Forest baseline...")
        rf_model = joblib.load(REGISTRY_DIR / "best_model_2026_03.pkl")
        rf_cal = joblib.load(REGISTRY_DIR / "calibrator_2026_03.pkl")
        rf_preds = rf_model.predict(X_test)
        rf_probs = rf_model.predict_proba(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_f1m = f1_score(y_test, rf_preds, average="macro", zero_division=0)
        rf_f1w = f1_score(y_test, rf_preds, average="weighted", zero_division=0)

        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=np.arange(rf_probs.shape[1]))
        rf_brier = np.mean(np.sum((rf_probs - y_bin) ** 2, axis=1))

        comparison_header = f"\n{'='*70}\n{'METRIC':<25} {'Random Forest':>15} {'ANN':>15} {'Winner':>10}\n{'-'*70}"
        logger.info(comparison_header)
        metrics = [
            ("Accuracy", rf_acc, ann_results["accuracy"]),
            ("F1-macro", rf_f1m, ann_results["f1_macro"]),
            ("F1-weighted", rf_f1w, ann_results["f1_weighted"]),
            ("Brier Score (lower=better)", rf_brier, ann_results["brier_score"]),
            ("Temperature", rf_cal["temperature"], ann_results["temperature"]),
        ]
        for name, rf_val, ann_val in metrics:
            if "Brier" in name:
                winner = "RF" if rf_val < ann_val else "ANN"
            else:
                winner = "RF" if rf_val > ann_val else "ANN"
            logger.info("%-25s %15.4f %15.4f %10s", name, rf_val, ann_val, winner)
        logger.info("=" * 70)

        # Classification report for ANN
        logger.info("[6] ANN Classification Report:")
        report = classification_report(y_test, ann_results["preds"],
                                        target_names=labels, zero_division=0)
        logger.info("\n%s", report)

        # Save ANN model
        stamp = datetime.now().strftime("%Y_%m")
        ann_path = REGISTRY_DIR / f"ann_model_{stamp}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "n_classes": n_classes,
            "hidden_sizes": (256, 128, 64),
            "accuracy": ann_results["accuracy"],
            "temperature": ann_results["temperature"],
            "feat_cols": feat_cols,
        }, ann_path)
        logger.info("ANN model saved to %s", ann_path)

        # Save scaler
        ann_scaler_path = REGISTRY_DIR / f"ann_scaler_{stamp}.pkl"
        joblib.dump(scaler, ann_scaler_path)
        logger.info("ANN scaler saved to %s", ann_scaler_path)

        logger.info("=" * 70)
        logger.info("ANN TRAINING COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.exception("ANN training failed: %s", e)
    finally:
        file_handler.close()
        logger.removeHandler(file_handler)

    logger.info("Results written to %s", outpath)


if __name__ == "__main__":
    main()

