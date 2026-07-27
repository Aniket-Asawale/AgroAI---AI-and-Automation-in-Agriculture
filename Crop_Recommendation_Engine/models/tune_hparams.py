"""
Optuna tuning for RF / XGBoost / LightGBM on the train+val split.

Objective = val_accuracy - overfit_penalty, where
    overfit_penalty = max(0, train_acc - val_acc - MAX_TRAIN_VAL_GAP)

This way tuning cannot win by raising train accuracy alone; it must keep the
train-val gap within the configured ceiling.

Writes model_registry/best_hparams.json when done. baseline_models.py picks it
up automatically on the next run.

Usage:
    python Crop_Recommendation_Engine/models/tune_hparams.py [--trials N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MAX_TRAIN_VAL_GAP, OPTUNA_TRIALS, RANDOM_STATE, REGISTRY_DIR,
)
from models.baseline_models import load_data

logger = logging.getLogger(__name__)


def _gap_penalised(train_acc: float, val_acc: float) -> float:
    gap = max(0.0, train_acc - val_acc - MAX_TRAIN_VAL_GAP)
    return val_acc - gap


def _tune_rf(trial, X_tr, y_tr, X_va, y_va) -> float:
    from sklearn.ensemble import RandomForestClassifier
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 800, step=100),
        "max_depth":        trial.suggest_int("max_depth", 8, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 4, 12),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 2, 8),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "max_samples":      trial.suggest_float("max_samples", 0.6, 0.9),
    }
    m = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=RANDOM_STATE,
        n_jobs=-1, **params,
    )
    m.fit(X_tr, y_tr)
    tr = m.score(X_tr, y_tr); va = m.score(X_va, y_va)
    return _gap_penalised(tr, va)


def _tune_xgb(trial, X_tr, y_tr, X_va, y_va) -> float:
    from xgboost import XGBClassifier
    n_classes = int(len(np.unique(y_tr)))
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 500, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 12),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
    }
    m = XGBClassifier(
        num_class=n_classes, objective="multi:softprob",
        eval_metric="mlogloss", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **params,
    )
    m.fit(X_tr, y_tr)
    tr = m.score(X_tr, y_tr); va = m.score(X_va, y_va)
    return _gap_penalised(tr, va)


def _tune_lgbm(trial, X_tr, y_tr, X_va, y_va) -> float:
    from lightgbm import LGBMClassifier
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 600, step=50),
        "max_depth":        trial.suggest_int("max_depth", 4, 8),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
        "learning_rate":    trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
    }
    m = LGBMClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1, subsample_freq=1, **params,
    )
    m.fit(X_tr, y_tr)
    tr = m.score(X_tr, y_tr); va = m.score(X_va, y_va)
    return _gap_penalised(tr, va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("Loading data...")
    X_tr, X_va, _, y_tr, y_va, _, _feat, _labels, _le = load_data()
    logger.info("Train=%d  Val=%d", len(X_tr), len(X_va))

    results = {}
    for name, obj in [
        ("rf",   _tune_rf),
        ("xgb",  _tune_xgb),
        ("lgbm", _tune_lgbm),
    ]:
        logger.info("=== Tuning %s (%d trials) ===", name.upper(), args.trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda t, obj=obj: obj(t, X_tr, y_tr, X_va, y_va),
            n_trials=args.trials, show_progress_bar=False,
        )
        results[name] = study.best_params
        logger.info("%s best val score (penalised) = %.4f  | params=%s",
                    name.upper(), study.best_value, study.best_params)

    out_path = REGISTRY_DIR / "best_hparams.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved \u2192 %s", out_path)


if __name__ == "__main__":
    main()
