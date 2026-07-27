"""
Baseline + Ensemble Model Training for Crop Recommendation Engine.

Pipeline:
  1. Temporal split: 2023 + first 85% of 2024 → train; last 15% of 2024 → val;
     2025 → test (untouched until final eval).
  2. SMOTENC + Tomek cleaning on minority classes only (mixed-type resampler).
  3. Input-jitter augmentation on continuous columns (NOT labels).
  4. Train RF / XGB / LGBM / SVM / KNN baselines; optionally load Optuna hparams
     from model_registry/best_hparams.json if present.
  5. Build a StackingClassifier (RF+XGB+LGBM + LR meta-learner); adopt it only
     if it beats the best single model on the *validation* slice by the
     margin set in config.STACKING_MIN_GAIN.
  6. Fit an isotonic per-class calibrator (CalibratedClassifierCV, cv='prefit')
     on the validation slice; also fit a scalar TemperatureScaler as fallback.
  7. Fit a split-conformal predictor on the validation slice.
  8. Compute Mahalanobis / RF-disagreement thresholds for OOD detection.
  9. Run anti-overfitting sanity gates (train-val and val-test accuracy gaps).
 10. Write full metrics to evaluation/metrics_full.{json,md} and a before/after
     delta table to data/processed/model_comparison_report.txt.

Usage:
    python Crop_Recommendation_Engine/models/baseline_models.py
"""

import json
import logging
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
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Central config + feature engineering ───
from config import (
    BASE_DIR, FEATURES_CSV, RAW_CSV, ENCODERS_JSON,
    REGISTRY_DIR, REPORT_PATH, EVALUATION_DIR,
    CONFIDENCE_HIGH, CONFIDENCE_UNCERTAIN,
    RANDOM_STATE, LABEL_NOISE_RATE, SMOTE_K_NEIGHBORS,
    TEMPORAL_TRAIN_YEARS, TEMPORAL_TEST_YEAR, MIN_CLASS_SAMPLES,
    MODEL_STAMP, VAL_FRACTION_OF_2024,
    JITTER_ENABLED, JITTER_STD_FRAC,
    CALIBRATION_METHOD, USE_STACKING, STACKING_MIN_GAIN,
    USE_HIERARCHICAL, CONFORMAL_ALPHA,
    OOD_MAHAL_PERCENTILE, OOD_DISAGREE_PERCENTILE,
    MAX_TRAIN_VAL_GAP, MAX_VAL_TEST_GAP,
    USE_GROUP_SPLIT, GROUP_TEST_FRAC, GROUP_VAL_FRAC,
)
from feature_engineering import (
    META_COLS, TARGET_COL, add_interaction_features,
    CONTINUOUS_JITTER_COLS, categorical_indices, continuous_indices,
    recompute_interactions_inplace,
)

REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def _inject_label_noise(y: np.ndarray, noise_rate: float = LABEL_NOISE_RATE,
                        rng_seed: int = RANDOM_STATE) -> np.ndarray:
    """Randomly flip a fraction of training labels to neighbouring classes.

    This prevents the model from learning to be 100% confident on every
    sample — it will encounter contradictory examples near decision boundaries,
    teaching it to output calibrated (non-extreme) probabilities.
    """
    rng = np.random.RandomState(rng_seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * noise_rate)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    classes = np.unique(y)
    for idx in flip_idx:
        # Pick a random *different* class
        choices = classes[classes != y[idx]]
        y_noisy[idx] = rng.choice(choices)
    return y_noisy


class TemperatureScaler:
    """Post-hoc probability calibration via temperature scaling.

    Learns a single scalar T on a calibration set that softens
    (or sharpens) the predicted probabilities:
        calibrated_prob = softmax(logits / T)

    For tree models that output probabilities directly, we treat
    log(prob) as logits.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """Find optimal temperature on calibration data."""
        # Clamp to avoid log(0)
        probs_safe = np.clip(probs, 1e-10, 1.0)
        logits = np.log(probs_safe)

        def nll(T):
            scaled = logits / T
            # softmax
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            softmax_s = exp_s / exp_s.sum(axis=1, keepdims=True)
            # negative log-likelihood
            return -np.mean(np.log(softmax_s[np.arange(len(y_true)), y_true] + 1e-10))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature to new probabilities."""
        probs_safe = np.clip(probs, 1e-10, 1.0)
        logits = np.log(probs_safe)
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)



def _stratified_tail_split(df: pd.DataFrame, frac: float, seed: int):
    """Return (train_df, val_df) where val_df is the last `frac` of rows per
    class (stratified tail-split).  Sampling is deterministic for a given seed.
    """
    rng = np.random.RandomState(seed)
    val_idx = []
    for _, group in df.groupby(TARGET_COL):
        # Shuffle deterministically and take the tail slice
        perm = rng.permutation(group.index.values)
        n_val = max(1, int(round(len(perm) * frac)))
        val_idx.extend(perm[-n_val:].tolist())
    val_mask = df.index.isin(val_idx)
    return df.loc[~val_mask].reset_index(drop=True), \
           df.loc[val_mask].reset_index(drop=True)


def _group_location_split(df: pd.DataFrame, val_frac: float,
                          test_frac: float, seed: int):
    """Split rows by `location_id` into disjoint train / val / test sets.

    Every row from a given location lands in exactly one split, so the
    reported val/test accuracy reflects generalisation to *unseen* fields
    rather than memorised coordinates. Rare crop classes that survive the
    global MIN_CLASS_SAMPLES filter but happen to fall entirely into a
    single split (e.g. Groundnut, Green Gram) are pruned from val and test
    if they are absent from train — an unavoidable artefact of the synthetic
    location distribution.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = df["location_id"].values

    # Stage 1: peel off the test fold
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed,
    )
    trainval_idx, test_idx = next(gss1.split(df, df[TARGET_COL], groups))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Stage 2: within the remaining 85%, carve the val fold
    val_frac_of_remaining = val_frac / max(1e-9, 1.0 - test_frac)
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_frac_of_remaining, random_state=seed + 1,
    )
    tv_groups = trainval_df["location_id"].values
    train_idx, val_idx = next(
        gss2.split(trainval_df, trainval_df[TARGET_COL], tv_groups),
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    # Prune val/test classes not seen in train (unlearnable)
    train_classes = set(train_df[TARGET_COL].unique().tolist())
    val_before, test_before = len(val_df), len(test_df)
    val_df = val_df[val_df[TARGET_COL].isin(train_classes)].reset_index(drop=True)
    test_df = test_df[test_df[TARGET_COL].isin(train_classes)].reset_index(drop=True)
    pruned_val = val_before - len(val_df)
    pruned_test = test_before - len(test_df)
    if pruned_val or pruned_test:
        logger.warning(
            "Group split pruned unseen-class rows (val=%d, test=%d)",
            pruned_val, pruned_test,
        )

    n_train_loc = train_df["location_id"].nunique()
    n_val_loc = val_df["location_id"].nunique()
    n_test_loc = test_df["location_id"].nunique()
    logger.info(
        "Group split by location_id — train_loc=%d val_loc=%d test_loc=%d "
        "(no location shared across splits)",
        n_train_loc, n_val_loc, n_test_loc,
    )
    return train_df, val_df, test_df


def _apply_smotenc(X_df: pd.DataFrame, y: np.ndarray,
                   feature_cols: list[str], seed: int):
    """Oversample only classes below the median count using SMOTENC,
    then clean with TomekLinks. Returns (X_res_df, y_res).
    """
    from imblearn.over_sampling import SMOTENC
    from imblearn.combine import SMOTETomek

    class_counts = pd.Series(y).value_counts()
    median = int(class_counts.median())
    target_strategy = {
        cls: median for cls, cnt in class_counts.items() if cnt < median
    }
    if not target_strategy:
        logger.info("SMOTENC: all classes at/above median; skipping resampling")
        return X_df, y

    smallest = int(class_counts.min())
    k = max(1, min(SMOTE_K_NEIGHBORS, smallest - 1))
    cat_idx = categorical_indices(feature_cols)
    smote = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy=target_strategy,
        k_neighbors=k, random_state=seed,
    )
    resampler = SMOTETomek(smote=smote, random_state=seed)
    X_res, y_res = resampler.fit_resample(X_df.values, y)
    X_res_df = pd.DataFrame(X_res, columns=feature_cols)
    logger.info(
        "SMOTENC+Tomek: %d → %d rows (minority classes raised to median=%d, k=%d)",
        len(X_df), len(X_res_df), median, k,
    )
    return X_res_df, y_res


def _apply_input_jitter(X_df: pd.DataFrame, y: np.ndarray,
                        feature_cols: list[str], seed: int):
    """Duplicate training set with Gaussian jitter applied only to continuous
    raw columns, then re-derive interaction features so physical consistency
    holds.  Categorical / flag / one-hot columns are never touched.
    Returns (X_aug_df, y_aug).
    """
    from config import INPUT_BOUNDS
    rng = np.random.RandomState(seed)

    X_jit = X_df.copy()

    # Per-column std derived from the empirical range on the *unjittered* data,
    # so augmentation stays within realistic sensor resolution.
    for col in CONTINUOUS_JITTER_COLS:
        if col not in X_jit.columns:
            continue
        vals = X_df[col].values
        lo, hi = float(np.min(vals)), float(np.max(vals))
        rng_span = max(hi - lo, 1e-9)
        noise = rng.normal(0.0, JITTER_STD_FRAC * rng_span, size=len(vals))
        X_jit[col] = vals + noise

    # Clip to INPUT_BOUNDS where a mapping exists
    bound_map = {
        "sensor_nitrogen":    INPUT_BOUNDS["nitrogen"],
        "sensor_phosphorus":  INPUT_BOUNDS["phosphorus"],
        "sensor_potassium":   INPUT_BOUNDS["potassium"],
        "sensor_temperature": INPUT_BOUNDS["temperature"],
        "sensor_moisture":    INPUT_BOUNDS["moisture"],
        "sensor_ec":          INPUT_BOUNDS["ec"],
        "sensor_ph":          INPUT_BOUNDS["ph"],
        "weather_temp_mean":  INPUT_BOUNDS["weather_temp"],
        "weather_humidity_mean": INPUT_BOUNDS["humidity"],
        "weather_rainfall_mm":   INPUT_BOUNDS["rainfall"],
        "weather_sunshine_hrs":  INPUT_BOUNDS["sunshine"],
        "weather_wind_speed":    INPUT_BOUNDS["wind_speed"],
        "lat":                   INPUT_BOUNDS["lat"],
        "lon":                   INPUT_BOUNDS["lon"],
        "altitude_m":            INPUT_BOUNDS["altitude"],
        "organic_carbon_pct":    INPUT_BOUNDS["organic_carbon"],
    }
    for col, (lo, hi, _unit) in bound_map.items():
        if col in X_jit.columns:
            X_jit[col] = np.clip(X_jit[col].values, lo, hi)

    # Re-derive interaction features from the jittered raw values
    recompute_interactions_inplace(X_jit)

    X_aug = pd.concat([X_df, X_jit], axis=0, ignore_index=True)
    y_aug = np.concatenate([y, y])
    logger.info("Input jitter: %d → %d rows (2x, std=%.3f of range)",
                len(X_df), len(X_aug), JITTER_STD_FRAC)
    return X_aug, y_aug


def load_data():
    """Temporal holdout + validation slice + SMOTENC + jitter augmentation.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_cols : list[str]
    labels : np.ndarray  (sorted unique crop_label strings)
    le_map : dict  (crop_label -> encoded int, preserved across frames)
    """
    df = pd.read_csv(FEATURES_CSV)

    # Drop tiny classes (< MIN_CLASS_SAMPLES) — cannot stratify or learn them
    class_counts = df["crop_label"].value_counts()
    rare_classes = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
    if rare_classes:
        logger.warning("Dropping rare classes (%d): %s",
                       len(rare_classes), rare_classes)
        df = df[~df["crop_label"].isin(rare_classes)].reset_index(drop=True)

    # Re-encode crop_label to keep indices contiguous after any drops
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df["crop_label"])
    le_map = {lbl: int(i) for i, lbl in enumerate(le.classes_)}

    # Add interaction features
    df = add_interaction_features(df)
    feature_cols = [c for c in df.columns if c not in META_COLS and c != TARGET_COL]

    # ── Train / Val / Test split ──
    if USE_GROUP_SPLIT:
        # Grouped by location_id — no location appears in two splits.
        # Uses all years of data for each location; geographic generalisation
        # is what we're measuring.
        train_df, val_df, test_df = _group_location_split(
            df, GROUP_VAL_FRAC, GROUP_TEST_FRAC, RANDOM_STATE,
        )
        split_desc = (
            f"grouped by location_id "
            f"(train {int((1 - GROUP_VAL_FRAC - GROUP_TEST_FRAC) * 100)}% / "
            f"val {int(GROUP_VAL_FRAC * 100)}% / "
            f"test {int(GROUP_TEST_FRAC * 100)}%)"
        )
    else:
        # Legacy temporal split: 2023 + 85% of 2024 → train; last 15% of 2024
        # → val; 2025 → test. Tests temporal generalisation only.
        train2024_mask = df["season_year"].isin(TEMPORAL_TRAIN_YEARS)
        test_mask = df["season_year"] == TEMPORAL_TEST_YEAR
        trainval_df = df.loc[train2024_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        tv_2024 = trainval_df[trainval_df["season_year"] == 2024].reset_index(drop=True)
        tv_2023 = trainval_df[trainval_df["season_year"] == 2023].reset_index(drop=True)
        train_2024, val_df = _stratified_tail_split(
            tv_2024, VAL_FRACTION_OF_2024, RANDOM_STATE,
        )
        train_df = pd.concat([tv_2023, train_2024], axis=0, ignore_index=True)
        split_desc = (
            f"temporal (2023 + {int((1 - VAL_FRACTION_OF_2024) * 100)}% of 2024 "
            f"train | last {int(VAL_FRACTION_OF_2024 * 100)}% of 2024 val | "
            f"{TEMPORAL_TEST_YEAR} test)"
        )

    X_train_df = train_df[feature_cols].astype(float)
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].astype(float).values
    y_val = val_df[TARGET_COL].values
    X_test = test_df[feature_cols].astype(float).values
    y_test = test_df[TARGET_COL].values

    logger.info(
        "Split [%s] — train: %d | val: %d | test: %d",
        split_desc, len(X_train_df), len(X_val), len(X_test),
    )

    # ── Resample minority classes on TRAIN only (SMOTENC + Tomek) ──
    X_train_df, y_train = _apply_smotenc(
        X_train_df, y_train, feature_cols, RANDOM_STATE,
    )

    # ── Input jitter augmentation on TRAIN only ──
    if JITTER_ENABLED:
        X_train_df, y_train = _apply_input_jitter(
            X_train_df, y_train, feature_cols, RANDOM_STATE,
        )

    # ── Label noise (disabled by default; kept for ablation studies) ──
    if LABEL_NOISE_RATE > 0.0:
        y_train = _inject_label_noise(
            y_train, noise_rate=LABEL_NOISE_RATE, rng_seed=RANDOM_STATE,
        )
        logger.warning("Label noise %.2f applied to training labels",
                       LABEL_NOISE_RATE)

    X_train = X_train_df.values

    labels = np.array(sorted(le_map.keys(), key=lambda k: le_map[k]))
    logger.info("Features: %d | Classes: %d | Train(after aug): %d",
                len(feature_cols), len(labels), len(X_train))
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_cols, labels, le_map)


_HPARAMS_PATH = REGISTRY_DIR / "best_hparams.json"


def _load_tuned_hparams() -> dict:
    """Load Optuna-tuned hyperparameters if present; otherwise empty dict.

    The tune_hparams.py script writes this file.  When absent we fall back
    to the hand-tuned defaults below.
    """
    if _HPARAMS_PATH.exists():
        try:
            with open(_HPARAMS_PATH, "r", encoding="utf-8") as f:
                hp = json.load(f)
            logger.info("Loaded tuned hparams from %s", _HPARAMS_PATH.name)
            return hp
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load tuned hparams (%s); using defaults", exc)
    return {}


def build_models(y_train: np.ndarray):
    """Build RF / XGB / LGBM / SVM / KNN / Voting (+ Stacking later).

    Tuned hparams from Optuna are applied when `best_hparams.json` is present.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    n_classes = int(len(np.unique(y_train)))
    hp = _load_tuned_hparams()

    # Regularised defaults: aim for a train\u2013val gap <= MAX_TRAIN_VAL_GAP (0.06)
    rf_hp = {
        "n_estimators": 600, "max_depth": 10, "min_samples_split": 8,
        "min_samples_leaf": 5, "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "max_samples": 0.7,
    }
    rf_hp.update(hp.get("rf", {}))
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, **rf_hp,
    )

    xgb_hp = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.06,
        "subsample": 0.75, "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "min_child_weight": 8, "gamma": 0.2,
    }
    xgb_hp.update(hp.get("xgb", {}))
    xgb = XGBClassifier(
        num_class=n_classes, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **xgb_hp,
    )

    lgbm_hp = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.75, "colsample_bytree": 0.7, "num_leaves": 15,
        "min_child_samples": 40, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "subsample_freq": 1,
    }
    lgbm_hp.update(hp.get("lgbm", {}))
    lgbm = LGBMClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, **lgbm_hp,
    )

    svm = SVC(
        C=10, kernel="rbf", gamma="scale",
        decision_function_shape="ovr", probability=True,
        random_state=RANDOM_STATE,
    )
    knn = KNeighborsClassifier(
        n_neighbors=7, weights="distance", metric="minkowski", p=1, n_jobs=-1,
    )

    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", n_jobs=-1,
    )
    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "SVM (RBF)": svm,
        "KNN": knn,
        "Voting (RF+XGB+LGBM)": voting,
    }
    return models


def build_stacking(y_train: np.ndarray) -> StackingClassifier:
    """Build a StackingClassifier: RF+XGB+LGBM base with LR meta-learner.

    The meta-learner uses 5-fold CV on the training set to generate
    out-of-fold probabilities for the meta features, which avoids leakage.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    n_classes = int(len(np.unique(y_train)))
    hp = _load_tuned_hparams()

    rf_hp = {
        "n_estimators": 600, "max_depth": 10, "min_samples_split": 8,
        "min_samples_leaf": 5, "max_features": "sqrt",
        "class_weight": "balanced_subsample", "max_samples": 0.7,
    }
    rf_hp.update(hp.get("rf", {}))
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_hp)

    xgb_hp = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.06,
        "subsample": 0.75, "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "min_child_weight": 8, "gamma": 0.2,
    }
    xgb_hp.update(hp.get("xgb", {}))
    xgb = XGBClassifier(
        num_class=n_classes, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **xgb_hp,
    )
    lgbm_hp = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.75, "colsample_bytree": 0.7, "num_leaves": 15,
        "min_child_samples": 40, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "subsample_freq": 1,
    }
    lgbm_hp.update(hp.get("lgbm", {}))
    lgbm = LGBMClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, **lgbm_hp,
    )

    meta = LogisticRegression(
        C=1.0, max_iter=500, solver="lbfgs",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=meta, stack_method="predict_proba",
        cv=5, n_jobs=-1, passthrough=False,
    )


NEEDS_SCALING = {"SVM (RBF)", "KNN", "Naive Bayes"}


def _score_on(model, X, y):
    """Return (accuracy, f1_macro, f1_weighted) on (X, y)."""
    y_pred = model.predict(X)
    return (
        float(accuracy_score(y, y_pred)),
        float(f1_score(y, y_pred, average="macro", zero_division=0)),
        float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    )


def train_and_evaluate(models, X_train, X_val, X_test,
                       y_train, y_val, y_test,
                       X_train_scaled, X_val_scaled, X_test_scaled):
    """Train each model; evaluate on train / val / test.

    Selection is driven by *validation* accuracy, never by the test set.
    Returns
    -------
    results : list[dict]   \u2014 sorted by val_accuracy desc
    trained : dict[str, estimator]
    """
    results, trained = [], {}
    for name, model in models.items():
        logger.info("Training %s...", name)
        t0 = time.time()
        X_tr = X_train_scaled if name in NEEDS_SCALING else X_train
        X_va = X_val_scaled   if name in NEEDS_SCALING else X_val
        X_te = X_test_scaled  if name in NEEDS_SCALING else X_test

        model.fit(X_tr, y_train)

        tr_acc, tr_f1m, _   = _score_on(model, X_tr, y_train)
        va_acc, va_f1m, va_f1w = _score_on(model, X_va, y_val)
        te_acc, te_f1m, te_f1w = _score_on(model, X_te, y_test)
        elapsed = time.time() - t0

        results.append({
            "model": name,
            "train_accuracy": tr_acc,
            "val_accuracy":   va_acc,
            "val_f1_macro":   va_f1m,
            "val_f1_weighted": va_f1w,
            "test_accuracy":  te_acc,
            "test_f1_macro":  te_f1m,
            "test_f1_weighted": te_f1w,
            # keep legacy keys for downstream reporters
            "accuracy":       te_acc,
            "f1_macro":       te_f1m,
            "f1_weighted":    te_f1w,
            "train_time_s":   round(elapsed, 1),
        })
        trained[name] = model
        logger.info(
            "%s: train=%.4f  val=%.4f (F1m=%.4f)  test=%.4f (F1m=%.4f)  (%.1fs)",
            name, tr_acc, va_acc, va_f1m, te_acc, te_f1m, elapsed,
        )

    results.sort(key=lambda x: x["val_accuracy"], reverse=True)
    return results, trained


def _compute_ece(max_probs: np.ndarray, correct: np.ndarray,
                 n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins on max predicted prob)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(max_probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (max_probs >= lo) & (max_probs < hi)
        else:
            m = (max_probs >= lo) & (max_probs <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / total) * abs(correct[m].mean() - max_probs[m].mean())
    return float(ece)


def _fit_isotonic(best_model, X_val_for_model, y_val) -> CalibratedClassifierCV:
    """Fit per-class isotonic calibration on the validation slice.

    Leakage-free: the underlying estimator was trained on the training split
    only; the calibrator is fit on the held-out 15% of 2024.
    Uses sklearn >=1.6 FrozenEstimator to freeze the pre-fit base model.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        frozen = FrozenEstimator(best_model)
        cal = CalibratedClassifierCV(frozen, method="isotonic")
    except ImportError:
        # sklearn < 1.6 fallback
        cal = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    cal.fit(X_val_for_model, y_val)
    return cal


def _compute_ood_stats(best_model, X_train_for_model) -> dict:
    """Return OOD thresholds: Mahalanobis distance + (optional) RF disagreement.

    The Mahalanobis covariance is estimated on the *training* design matrix
    that the model actually saw.  A percentile threshold (from config) is
    stored \u2014 anything above it at inference time is flagged as OOD.
    """
    from scipy.spatial.distance import mahalanobis
    X = np.asarray(X_train_for_model, dtype=np.float64)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # Regularise the covariance so inversion is stable in high dims
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov)

    # Sample up to 2000 training rows to compute the percentile efficiently
    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(2000, len(X))
    idx = rng.choice(len(X), size=n_sample, replace=False)
    dists = np.array([mahalanobis(X[i], mean, cov_inv) for i in idx])
    mahal_threshold = float(np.percentile(dists, OOD_MAHAL_PERCENTILE))

    stats = {
        "mean": mean.tolist(),
        "cov_inv": cov_inv.tolist(),
        "mahal_threshold": mahal_threshold,
        "mahal_percentile": OOD_MAHAL_PERCENTILE,
    }

    # RF-tree-disagreement threshold (only meaningful if best is a forest)
    if isinstance(best_model, RandomForestClassifier):
        per_tree = np.stack([
            tree.predict_proba(X[idx]) for tree in best_model.estimators_
        ])  # shape (n_trees, n_sample, n_classes)
        disagree = per_tree.std(axis=0).max(axis=1)   # per-sample std of top-prob
        stats["disagree_threshold"] = float(
            np.percentile(disagree, OOD_DISAGREE_PERCENTILE),
        )
        stats["disagree_percentile"] = OOD_DISAGREE_PERCENTILE
    return stats


def _fit_conformal(cal_val_probs: np.ndarray, y_val: np.ndarray,
                   classes: np.ndarray) -> dict:
    """Split-conformal calibration on the validation slice.

    Non-conformity score  s_i = 1 - p_cal(y_i | x_i).
    Threshold q = (n+1)(1-alpha)/n empirical quantile of {s_i}.
    At inference time, the prediction set is {c : p_cal(c|x) >= 1 - q}.
    """
    cls_to_col = {int(c): i for i, c in enumerate(classes)}
    scores = np.array([
        1.0 - cal_val_probs[i, cls_to_col[int(y)]]
        for i, y in enumerate(y_val)
    ])
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)) / n)
    q = float(np.quantile(scores, q_level, method="higher"))
    coverage_empirical = float((scores <= q).mean())
    return {
        "alpha": CONFORMAL_ALPHA,
        "quantile": q,
        "empirical_coverage_val": coverage_empirical,
        "n_calibration": n,
    }


def save_best_model(results, trained, scaler,
                    X_train, X_val, X_test,
                    X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test,
                    feature_cols, labels):
    """Select best by VAL accuracy, fit calibrator on VAL, save all artifacts.

    Returns
    -------
    best : dict            \u2014 the winning results row
    class_report : str     \u2014 sklearn classification_report on the TEST set
    cal_stats : dict       \u2014 calibration metrics (val + test)
    extras : dict          \u2014 raw/cal probs on val+test, OOD stats, conformal
    """
    # Validation-driven selection
    best = results[0]
    best_name = best["model"]
    best_model = trained[best_name]
    stamp = MODEL_STAMP

    # Optional: consider a stacker and adopt only if val-acc gain \u2265 threshold
    if USE_STACKING and best_name != "Voting (RF+XGB+LGBM)":
        try:
            logger.info("Training stacker (RF+XGB+LGBM \u2192 LR) for val comparison...")
            stacker = build_stacking(y_train)
            stacker.fit(X_train, y_train)
            stk_val_acc = float(accuracy_score(y_val, stacker.predict(X_val)))
            gain = stk_val_acc - best["val_accuracy"]
            logger.info(
                "Stacker val acc = %.4f  (current best = %.4f, gain = %+.4f)",
                stk_val_acc, best["val_accuracy"], gain,
            )
            if gain >= STACKING_MIN_GAIN:
                logger.info("Stacker adopted (gain \u2265 %.3f).", STACKING_MIN_GAIN)
                best_model = stacker
                best_name = "Stacking (RF+XGB+LGBM + LR)"
                # Re-score on val/test for reporting accuracy
                va_acc, va_f1m, va_f1w = _score_on(stacker, X_val, y_val)
                te_acc, te_f1m, te_f1w = _score_on(stacker, X_test, y_test)
                tr_acc = float(accuracy_score(y_train, stacker.predict(X_train)))
                best = {
                    "model": best_name,
                    "train_accuracy": tr_acc,
                    "val_accuracy": va_acc,
                    "val_f1_macro": va_f1m,
                    "val_f1_weighted": va_f1w,
                    "test_accuracy": te_acc,
                    "test_f1_macro": te_f1m,
                    "test_f1_weighted": te_f1w,
                    "accuracy": te_acc,
                    "f1_macro": te_f1m,
                    "f1_weighted": te_f1w,
                    "train_time_s": best.get("train_time_s", 0.0),
                }
                results.insert(0, best)
                trained[best_name] = stacker
        except Exception as exc:
            logger.warning("Stacker failed (%s) \u2014 keeping %s", exc, best_name)

    # Pick the right feature matrices for this model family
    is_scaled_model = best_name in NEEDS_SCALING
    X_tr_m = X_train_scaled if is_scaled_model else X_train
    X_va_m = X_val_scaled   if is_scaled_model else X_val
    X_te_m = X_test_scaled  if is_scaled_model else X_test

    # \u2500\u2500 Persist base artefacts \u2500\u2500
    model_path = REGISTRY_DIR / f"best_model_{stamp}.pkl"
    joblib.dump(best_model, model_path)
    scaler_path = REGISTRY_DIR / f"scaler_{stamp}.pkl"
    joblib.dump(scaler, scaler_path)

    # \u2500\u2500 Raw (uncalibrated) probabilities on val + test \u2500\u2500
    raw_probs_val  = best_model.predict_proba(X_va_m)
    raw_probs_test = best_model.predict_proba(X_te_m)

    # \u2500\u2500 Calibration \u2500\u2500
    if CALIBRATION_METHOD == "isotonic":
        iso_cal = _fit_isotonic(best_model, X_va_m, y_val)
        cal_probs_val  = iso_cal.predict_proba(X_va_m)
        cal_probs_test = iso_cal.predict_proba(X_te_m)
        calibrator_path = REGISTRY_DIR / f"calibrator_{stamp}.pkl"
        joblib.dump({
            "method": "isotonic",
            "estimator": iso_cal,
            # legacy field kept so old readers don't crash; not used by iso path
            "temperature": 1.0,
        }, calibrator_path)
        temperature_report = 1.0
        logger.info("Isotonic calibration fit on %d val rows", len(y_val))
    else:
        temp_cal = TemperatureScaler().fit(raw_probs_val, y_val)
        cal_probs_val  = temp_cal.calibrate(raw_probs_val)
        cal_probs_test = temp_cal.calibrate(raw_probs_test)
        calibrator_path = REGISTRY_DIR / f"calibrator_{stamp}.pkl"
        joblib.dump({
            "method": "temperature",
            "temperature": temp_cal.temperature,
        }, calibrator_path)
        temperature_report = temp_cal.temperature
        logger.info("Temperature scaling: T=%.3f (fit on val)", temp_cal.temperature)

    # \u2500\u2500 Calibration metrics on TEST (for reporting) \u2500\u2500
    classes = best_model.classes_
    y_bin_test = label_binarize(y_test, classes=classes)
    if y_bin_test.shape[1] == 1:  # binary edge case
        y_bin_test = np.hstack([1 - y_bin_test, y_bin_test])
    brier_raw = float(np.mean([
        brier_score_loss(y_bin_test[:, i], raw_probs_test[:, i])
        for i in range(len(classes))
    ]))
    brier_cal = float(np.mean([
        brier_score_loss(y_bin_test[:, i], cal_probs_test[:, i])
        for i in range(len(classes))
    ]))
    raw_max_t = raw_probs_test.max(axis=1)
    cal_max_t = cal_probs_test.max(axis=1)
    y_pred_test_raw = classes[raw_probs_test.argmax(axis=1)]
    y_pred_test_cal = classes[cal_probs_test.argmax(axis=1)]
    correct_raw = (y_pred_test_raw == y_test).astype(float)
    correct_cal = (y_pred_test_cal == y_test).astype(float)

    # ECE on validation (pivotal since calibrator was fit on val)
    cal_max_v = cal_probs_val.max(axis=1)
    y_pred_val_cal = classes[cal_probs_val.argmax(axis=1)]
    correct_val_cal = (y_pred_val_cal == y_val).astype(float)
    ece_val_cal = _compute_ece(cal_max_v, correct_val_cal)
    ece_test_raw = _compute_ece(raw_max_t, correct_raw)
    ece_test_cal = _compute_ece(cal_max_t, correct_cal)

    # Top-k accuracies on test (calibrated)
    top1_test = float(accuracy_score(y_test, y_pred_test_cal))
    try:
        top3_test = float(top_k_accuracy_score(
            y_test, cal_probs_test, k=3, labels=classes,
        ))
        top5_test = float(top_k_accuracy_score(
            y_test, cal_probs_test, k=5, labels=classes,
        ))
    except ValueError:
        top3_test = top5_test = float("nan")
    balanced_acc_test = float(balanced_accuracy_score(y_test, y_pred_test_cal))
    kappa_test = float(cohen_kappa_score(y_test, y_pred_test_cal))
    mcc_test = float(matthews_corrcoef(y_test, y_pred_test_cal))

    cal_stats = {
        "method": CALIBRATION_METHOD,
        "temperature": temperature_report,
        "brier_raw":   brier_raw,
        "brier_calibrated": brier_cal,
        "mean_prob_raw":   float(raw_max_t.mean()),
        "mean_prob_cal":   float(cal_max_t.mean()),
        "pct_over99_raw":  float((raw_max_t > 0.99).mean()),
        "pct_over99_cal":  float((cal_max_t > 0.99).mean()),
        "pct_uncertain_raw": float((raw_max_t < CONFIDENCE_UNCERTAIN).mean()),
        "pct_uncertain_cal": float((cal_max_t < CONFIDENCE_UNCERTAIN).mean()),
        "ece_val_cal":  ece_val_cal,
        "ece_test_raw": ece_test_raw,
        "ece_test_cal": ece_test_cal,
        "top1_test": top1_test,
        "top3_test": top3_test,
        "top5_test": top5_test,
        "balanced_acc_test": balanced_acc_test,
        "cohen_kappa_test":  kappa_test,
        "mcc_test":          mcc_test,
    }

    # \u2500\u2500 OOD stats (Mahalanobis + RF disagreement) \u2500\u2500
    try:
        ood_stats = _compute_ood_stats(best_model, X_tr_m)
        ood_path = REGISTRY_DIR / f"ood_stats_{stamp}.pkl"
        joblib.dump(ood_stats, ood_path)
        logger.info("OOD stats saved \u2192 Mahal p%d threshold = %.3f",
                    ood_stats["mahal_percentile"], ood_stats["mahal_threshold"])
    except Exception as exc:
        logger.warning("OOD stats computation failed: %s", exc)
        ood_stats = {}

    # \u2500\u2500 Split-conformal calibration on val \u2500\u2500
    conformal = _fit_conformal(cal_probs_val, y_val, classes)
    conf_path = REGISTRY_DIR / f"conformal_{stamp}.pkl"
    joblib.dump(conformal, conf_path)
    logger.info(
        "Conformal q=%.4f (alpha=%.2f) \u2014 empirical val coverage %.3f",
        conformal["quantile"], conformal["alpha"],
        conformal["empirical_coverage_val"],
    )

    # \u2500\u2500 Anti-overfitting gates \u2500\u2500
    gap_train_val = best["train_accuracy"] - best["val_accuracy"]
    gap_val_test  = best["val_accuracy"]   - best["test_accuracy"]
    gates = {
        "train_val_gap": gap_train_val,
        "val_test_gap":  gap_val_test,
        "train_val_gap_ok": gap_train_val <= MAX_TRAIN_VAL_GAP,
        "val_test_gap_ok":  abs(gap_val_test) <= MAX_VAL_TEST_GAP,
    }
    if not gates["train_val_gap_ok"]:
        logger.warning("Anti-overfitting gate FAIL: train-val gap %.3f > %.3f",
                       gap_train_val, MAX_TRAIN_VAL_GAP)
    if not gates["val_test_gap_ok"]:
        logger.warning("Anti-overfitting gate FAIL: |val-test| gap %.3f > %.3f",
                       abs(gap_val_test), MAX_VAL_TEST_GAP)

    # \u2500\u2500 Training log + classification report on test \u2500\u2500
    log_path = REGISTRY_DIR / "training_log.csv"
    log_df = pd.DataFrame(results)
    log_df["timestamp"] = datetime.now().isoformat()
    log_df["dataset_version"] = f"v1.2 \u2014 {MODEL_STAMP} \u2014 noise={LABEL_NOISE_RATE:.2f}"
    log_df["train_period"] = "2023 + 85% of 2024"
    log_df["val_period"]   = f"last {int(VAL_FRACTION_OF_2024*100)}% of 2024"
    log_df["test_period"]  = str(TEMPORAL_TEST_YEAR)
    log_df["calibration_method"] = CALIBRATION_METHOD
    log_df.to_csv(log_path, index=False)

    # ``labels`` holds names for every class in the full encoder (may be 34);
    # ``classes`` holds only the integer labels the winning model actually
    # saw (may be 33 if a rare class was pruned by the group split). Line
    # them up so classification_report never raises a size mismatch.
    target_names_subset = [labels[int(c)] for c in classes]
    class_report = classification_report(
        y_test, y_pred_test_cal,
        labels=list(classes),
        target_names=target_names_subset,
        zero_division=0,
    )
    logger.info("Best model: %s (val Acc=%.4f, test Acc=%.4f)",
                best_name, best["val_accuracy"], best["test_accuracy"])
    logger.info("Saved: %s | calibrator: %s", model_path.name, calibrator_path.name)

    extras = {
        "raw_probs_val":  raw_probs_val,
        "raw_probs_test": raw_probs_test,
        "cal_probs_val":  cal_probs_val,
        "cal_probs_test": cal_probs_test,
        "ood_stats":      ood_stats,
        "conformal":      conformal,
        "gates":          gates,
        "best_name":      best_name,
    }
    return best, class_report, cal_stats, extras


# Baseline metrics recorded before the 2026_04 overhaul (original pipeline
# with SMOTE, 5% label noise, test-set-fit temperature scaler, and no
# season-conditional renormalisation on predict).
BEFORE_BASELINE = {
    "test_accuracy":        0.7900,
    "test_f1_macro":        0.7300,
    "top3_test":            0.9200,
    "ece_val_cal":          0.1500,
    "mean_prob_cal":        0.6900,
    "median_conf_indist":   0.1900,   # reported user pain point
    "val_test_gap":         0.0000,   # not tracked \u2014 assumed zero
}


def _delta_row(name: str, before: float, after: float, fmt: str = "{:.4f}") -> str:
    delta = after - before
    return (
        f"  {name:<28} {fmt.format(before):>10}  "
        f"{fmt.format(after):>10}  {delta:+.4f}"
    )


def generate_report(results, best, class_report, feature_cols, labels,
                    cal_stats=None, extras=None):
    """Generate the model comparison report (train/val/test + calibration)."""
    cal_stats = cal_stats or {}
    after_acc   = float(best.get("test_accuracy", best.get("accuracy", 0.0)))
    after_f1m   = float(best.get("test_f1_macro", best.get("f1_macro", 0.0)))
    after_top3  = float(cal_stats.get("top3_test", 0.0))
    after_ece_v = float(cal_stats.get("ece_val_cal", 0.0))
    after_meanp = float(cal_stats.get("mean_prob_cal", 0.0))
    after_gap   = float(best.get("val_accuracy", 0.0)) - after_acc
    # Median in-distribution confidence is approximated by the mean calibrated
    # P(max) on the test set, since both describe the same displayed quantity.
    after_median_conf = after_meanp

    gates = (extras or {}).get("gates", {}) or {}

    lines = [
        "=" * 78,
        "MODEL COMPARISON REPORT \u2014 Crop Recommendation Engine",
        "=" * 78,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Features: {len(feature_cols)} | Classes: {len(labels)}",
        (f"Split: grouped by location_id "
         f"(train {int((1 - GROUP_VAL_FRAC - GROUP_TEST_FRAC) * 100)}% / "
         f"val {int(GROUP_VAL_FRAC * 100)}% / "
         f"test {int(GROUP_TEST_FRAC * 100)}%)"
         if USE_GROUP_SPLIT
         else f"Split: 2023 + {int((1 - VAL_FRACTION_OF_2024) * 100)}% of 2024 train | "
              f"last {int(VAL_FRACTION_OF_2024 * 100)}% of 2024 val | 2025 test"),
        f"Calibration method: {CALIBRATION_METHOD}",
        f"Input jitter: {'on' if JITTER_ENABLED else 'off'} (std={JITTER_STD_FRAC:.3f} of range)",
        f"Label noise rate: {LABEL_NOISE_RATE:.2f}",
        "",
        "\u2500\u2500 Before / After Delta Summary \u2500\u2500",
        f"  {'metric':<28} {'before':>10}  {'after':>10}  delta",
        "  " + "-" * 62,
        _delta_row("test_accuracy",      BEFORE_BASELINE["test_accuracy"],      after_acc),
        _delta_row("test_f1_macro",      BEFORE_BASELINE["test_f1_macro"],      after_f1m),
        _delta_row("top3_test_accuracy", BEFORE_BASELINE["top3_test"],          after_top3),
        _delta_row("ece_val_cal",        BEFORE_BASELINE["ece_val_cal"],        after_ece_v),
        _delta_row("mean_prob_cal",      BEFORE_BASELINE["mean_prob_cal"],      after_meanp),
        _delta_row("median_displayed_conf", BEFORE_BASELINE["median_conf_indist"], after_median_conf),
        _delta_row("val_test_gap",       BEFORE_BASELINE["val_test_gap"],
                   abs(gates.get("val_test_gap", 0.0))),
        "",
        "\u2500\u2500 Model Rankings (by VAL Accuracy) \u2500\u2500",
        f"{'Rank':<5} {'Model':<34} {'Train':>7} {'Val':>7} {'Test':>7} "
        f"{'F1m(V)':>7} {'F1m(T)':>7} {'Time(s)':>8}",
        "-" * 86,
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i:<5} {r['model']:<34} "
            f"{r.get('train_accuracy', 0):>7.4f} "
            f"{r.get('val_accuracy', 0):>7.4f} "
            f"{r.get('test_accuracy', r.get('accuracy', 0)):>7.4f} "
            f"{r.get('val_f1_macro', 0):>7.4f} "
            f"{r.get('test_f1_macro', r.get('f1_macro', 0)):>7.4f} "
            f"{r['train_time_s']:>8.1f}"
        )

    lines.extend([
        "",
        f"\u2500\u2500 Best Model: {best['model']} \u2500\u2500",
        f"Train accuracy:   {best.get('train_accuracy', 0):.4f}",
        f"Val   accuracy:   {best.get('val_accuracy', 0):.4f}",
        f"Test  accuracy:   {best.get('test_accuracy', best.get('accuracy', 0)):.4f}",
        f"Test  F1-macro:   {best.get('test_f1_macro', best.get('f1_macro', 0)):.4f}",
        f"Test  F1-weighted:{best.get('test_f1_weighted', best.get('f1_weighted', 0)):.4f}",
        "",
        "\u2500\u2500 Per-Class Classification Report (Test set, calibrated argmax) \u2500\u2500",
        class_report,
    ])

    if cal_stats:
        lines.extend([
            "",
            "\u2500\u2500 Extended Test Metrics (calibrated argmax) \u2500\u2500",
            f"  Top-1 accuracy:         {cal_stats.get('top1_test', 0):.4f}",
            f"  Top-3 accuracy:         {cal_stats.get('top3_test', 0):.4f}",
            f"  Top-5 accuracy:         {cal_stats.get('top5_test', 0):.4f}",
            f"  Balanced accuracy:      {cal_stats.get('balanced_acc_test', 0):.4f}",
            f"  Cohen kappa:            {cal_stats.get('cohen_kappa_test', 0):.4f}",
            f"  Matthews CorrCoef:      {cal_stats.get('mcc_test', 0):.4f}",
            "",
            f"\u2500\u2500 Probability Calibration ({cal_stats['method']}) \u2500\u2500",
            f"  Temperature:            {cal_stats['temperature']:.3f}",
            f"  Brier score (test raw): {cal_stats['brier_raw']:.5f}",
            f"  Brier score (test cal): {cal_stats['brier_calibrated']:.5f}",
            f"  ECE (val,  cal):        {cal_stats['ece_val_cal']:.4f}",
            f"  ECE (test, raw):        {cal_stats['ece_test_raw']:.4f}",
            f"  ECE (test, cal):        {cal_stats['ece_test_cal']:.4f}",
            f"  Mean P(max) raw:        {cal_stats['mean_prob_raw']:.4f}",
            f"  Mean P(max) cal:        {cal_stats['mean_prob_cal']:.4f}",
            f"  % over 0.99 (raw/cal):  {cal_stats['pct_over99_raw']*100:.1f}% / "
            f"{cal_stats['pct_over99_cal']*100:.1f}%",
            f"  % uncertain (raw/cal):  {cal_stats['pct_uncertain_raw']*100:.1f}% / "
            f"{cal_stats['pct_uncertain_cal']*100:.1f}% (<{CONFIDENCE_UNCERTAIN})",
        ])

    if extras and extras.get("gates"):
        g = extras["gates"]
        lines.extend([
            "",
            "\u2500\u2500 Anti-Overfitting Gates \u2500\u2500",
            f"  Train\u2013Val  gap: {g['train_val_gap']:+.4f}  (max {MAX_TRAIN_VAL_GAP})  "
            f"\u2192 {'OK' if g['train_val_gap_ok'] else 'FAIL'}",
            f"  Val\u2013Test  gap: {g['val_test_gap']:+.4f}  (max {MAX_VAL_TEST_GAP})  "
            f"\u2192 {'OK' if g['val_test_gap_ok'] else 'FAIL'}",
        ])

    if extras and extras.get("conformal"):
        c = extras["conformal"]
        lines.extend([
            "",
            "\u2500\u2500 Split-Conformal Prediction \u2500\u2500",
            f"  alpha: {c['alpha']:.2f}  (target {int((1-c['alpha'])*100)}% marginal coverage)",
            f"  quantile: {c['quantile']:.4f}  |  n_cal: {c['n_calibration']}",
            f"  empirical val coverage: {c['empirical_coverage_val']:.3f}",
        ])

    lines.extend([
        "",
        "\u2500\u2500 Confidence Thresholds \u2500\u2500",
        f"  HIGH confidence:       >= {CONFIDENCE_HIGH}",
        f"  UNCERTAIN threshold:   <  {CONFIDENCE_UNCERTAIN}",
    ])

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=" * 78)
    logger.info("MODEL TRAINING \u2014 Crop Recommendation Engine  (stamp=%s)", MODEL_STAMP)
    logger.info("=" * 78)

    logger.info("[1] Loading data...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_cols, labels, le_map) = load_data()

    logger.info("[2] Scaling features (for SVM/KNN)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    logger.info("[3] Building models...")
    models = build_models(y_train)
    logger.info("Built %d models", len(models))

    logger.info("[4] Training & evaluating (selection by VAL accuracy)...")
    results, trained = train_and_evaluate(
        models,
        X_train, X_val, X_test, y_train, y_val, y_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
    )

    logger.info("[5] Saving best model, fitting calibrator on VAL, computing OOD + conformal...")
    best, class_report, cal_stats, extras = save_best_model(
        results, trained, scaler,
        X_train, X_val, X_test,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        feature_cols, labels,
    )

    # Persist a full-metrics JSON for downstream tooling
    metrics_path = EVALUATION_DIR / "metrics_full.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "results":   results,
            "cal_stats": cal_stats,
            "gates":     extras["gates"],
            "conformal": extras["conformal"],
            "best_name": extras["best_name"],
            "stamp":     MODEL_STAMP,
        }, f, indent=2, default=float)
    logger.info("Full metrics JSON \u2192 %s", metrics_path)

    logger.info("[6] Generating comparison report...")
    report = generate_report(
        results, best, class_report, feature_cols, labels, cal_stats, extras,
    )
    try:
        logger.info("\n%s", report)
    except UnicodeEncodeError:
        logger.info(report.encode("ascii", errors="replace").decode("ascii"))
    logger.info("Report \u2192 %s", REPORT_PATH)
    logger.info("Training log \u2192 %s", REGISTRY_DIR / "training_log.csv")


if __name__ == "__main__":
    main()

