"""
Top-3 Crop Recommendation Inference Module.

Loads the trained Random Forest model, scaler, and temperature calibrator.
Accepts raw sensor/weather/soil inputs and returns a structured recommendation
with top-3 crops, calibrated confidence, confidence flags, and advisory notes.

Usage:
    from models.inference import CropRecommender
    recommender = CropRecommender()
    result = recommender.predict(
        nitrogen=110, phosphorus=55, potassium=70,
        temperature=26, moisture=55, ec=1400, ph=6.0,
        weather_temp=27, humidity=75, rainfall=900,
        sunshine=4.5, wind_speed=8,
        lat=20.93, lon=77.75, altitude=343,
        organic_carbon=0.67,
        soil_type="Black (Regur)", soil_texture="Clay Loam",
        drainage="Moderate", agro_zone="Vidarbha",
        season="Kharif", month=7,
    )
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path for central imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_engineering import META_COLS, TARGET_COL, add_interaction_features

# ─── Constants (from central config) ───
from config import (
    BASE_DIR, REGISTRY_DIR, FEATURES_CSV, ENCODERS_JSON,
    CONFIDENCE_HIGH as HIGH_THRESHOLD,
    CONFIDENCE_LOW as MEDIUM_THRESHOLD,
    MH_LAT_RANGE, MH_LON_RANGE,
    SOIL_DRAINAGE_MAP, OOD_PROB_THRESHOLD,
    VALID_SEASONS, VALID_SOIL_TYPES, INPUT_BOUNDS,
    MODEL_STAMP,
)




class CropRecommender:
    """Production-ready crop recommendation engine.

    Loads model artifacts once, then provides fast inference
    with calibrated top-3 predictions and confidence flags.
    """

    def __init__(self, model_stamp: Optional[str] = None):
        """Load model, scaler, calibrator, encoders, and feature column order.

        If model_stamp is None, auto-discovers the latest model in model_registry
        by sorting best_model_*.pkl files and picking the most recent.

        Per-artifact resolution rules:
          * model / scaler / ood_stats / conformal — losslessly lzma-compressed,
            so prefer the *_compressed.pkl sibling when present.
          * calibrator — the *_compressed.pkl is a *slim* approximation (per-fold
            isotonic threshold arrays only, no wrapped estimator). It is a small
            quality loss vs. the original CalibratedClassifierCV, so we prefer
            the original calibrator_<stamp>.pkl when available and only fall
            back to the slim file (the GitHub/Streamlit deployment case).
        """
        if model_stamp is None:
            model_stamp = self._discover_latest_stamp()
            logger.info("Auto-discovered model stamp: %s", model_stamp)

        self.model_stamp = model_stamp

        def _prefer_compressed(name: str) -> Path:
            """Return the *_compressed.pkl path if present, else the plain one."""
            compressed = REGISTRY_DIR / f"{name}_{model_stamp}_compressed.pkl"
            original   = REGISTRY_DIR / f"{name}_{model_stamp}.pkl"
            return compressed if compressed.exists() else original

        def _prefer_original(name: str) -> Path:
            """Return the plain *.pkl path if present, else the *_compressed.pkl one."""
            compressed = REGISTRY_DIR / f"{name}_{model_stamp}_compressed.pkl"
            original   = REGISTRY_DIR / f"{name}_{model_stamp}.pkl"
            return original if original.exists() else compressed

        model_path = _prefer_compressed("best_model")
        logger.info("Loading model from %s", model_path.name)
        self.model = joblib.load(model_path)

        scaler_path = _prefer_compressed("scaler")
        self.scaler = joblib.load(scaler_path)

        # ── Calibrator: prefer the FULL original (slim is a fallback only) ──
        cal_path = _prefer_original("calibrator")
        cal_data = joblib.load(cal_path)
        logger.info("Loading calibrator from %s", cal_path.name)

        self.calibration_method = cal_data.get("method", "temperature")
        self.temperature = float(cal_data.get("temperature", 1.0))

        # Full CalibratedClassifierCV (original format)
        self.iso_calibrator = cal_data.get("estimator", None)

        # Slim isotonic maps (compressed format — list of fold→class threshold arrays)
        # Shape: iso_maps[fold][class_idx] = {X_thresholds_, y_thresholds_, increasing_}
        self.iso_maps  = cal_data.get("iso_maps", None)
        self.iso_classes = cal_data.get("classes_", None)

        logger.info(
            "Loaded model=%s  calibration=%s  T=%.3f  slim_iso=%s",
            model_stamp, self.calibration_method, self.temperature,
            self.iso_maps is not None,
        )

        # ── Optional OOD stats (Mahalanobis + RF disagreement) ──
        ood_path = _prefer_compressed("ood_stats")
        self.ood_stats = joblib.load(ood_path) if ood_path.exists() else None
        if self.ood_stats:
            logger.info(
                "Loaded OOD stats - Mahal p%d threshold %.3f",
                self.ood_stats.get("mahal_percentile", 99),
                self.ood_stats.get("mahal_threshold", 0.0),
            )

        # ── Optional conformal quantile ──
        conf_path = _prefer_compressed("conformal")
        self.conformal = joblib.load(conf_path) if conf_path.exists() else None

        with open(ENCODERS_JSON) as f:
            self.encoders = json.load(f)

        # Reverse map: encoded int -> crop name (full training encoder; may
        # contain classes the model never saw if the group split pruned any).
        self.crop_labels = {v: k for k, v in self.encoders["crop_label"].items()}

        # Column-index -> crop name, aligned with self.model.classes_.
        # ``predict_proba`` returns columns in that order, so we use THIS map
        # whenever indexing into a probability vector. ``crop_labels`` stays
        # as the authoritative encoder lookup for reverse mappings.
        model_classes = getattr(self.model, "classes_", None)
        if model_classes is not None:
            self.col_to_crop = {
                i: self.crop_labels.get(int(c), f"Unknown_{int(c)}")
                for i, c in enumerate(model_classes)
            }
        else:
            self.col_to_crop = dict(self.crop_labels)
        self.n_model_classes = len(self.col_to_crop)

        # Get feature column order from training data
        df_ref = pd.read_csv(FEATURES_CSV, nrows=5)
        df_ref = add_interaction_features(df_ref)
        self.feat_cols = [c for c in df_ref.columns
                         if c not in META_COLS and c != TARGET_COL]
        logger.debug("Feature columns (%d): %s", len(self.feat_cols), self.feat_cols[:5])

    @staticmethod
    def _discover_latest_stamp() -> str:
        """Auto-discover the latest model stamp from model_registry.

        Strips the ``_compressed`` suffix and any OneDrive ``_copy``
        marker so the stamp stays as ``YYYY_MM`` regardless of which
        variant is actually present on disk.  The suffix is resolved
        transparently in ``__init__`` per artifact.
        """
        stamps = set()
        for f in REGISTRY_DIR.glob("best_model_*.pkl"):
            stem = f.stem.replace("best_model_", "")
            if " copy" in stem:
                continue
            if stem.endswith("_compressed"):
                stem = stem[: -len("_compressed")]
            stamps.add(stem)
        if not stamps:
            raise FileNotFoundError(
                f"No model files found in {REGISTRY_DIR}. "
                "Run baseline_models.py first to train a model."
            )
        # Latest by lexical order: works for YYYY_MM stamps.
        return sorted(stamps)[-1]

    def _calibrate(self, raw_probs: np.ndarray, X_for_model=None) -> np.ndarray:
        """Apply the loaded calibrator.

        Three paths (in priority order):
          1. Full CalibratedClassifierCV (original large calibrator file).
          2. Slim isotonic maps (compressed calibrator — threshold arrays only).
          3. Temperature scaling (legacy / fallback).
        """
        # Path 1: full CalibratedClassifierCV
        if (self.calibration_method == "isotonic"
                and self.iso_calibrator is not None
                and X_for_model is not None):
            return self.iso_calibrator.predict_proba(X_for_model)

        # Path 2: slim isotonic maps (compressed calibrator format)
        # iso_maps[fold][class_idx] = {X_thresholds_, y_thresholds_, increasing_}
        # We average across folds and apply np.interp per class.
        if (self.calibration_method == "isotonic"
                and self.iso_maps is not None):
            n_classes = raw_probs.shape[1]
            cal = np.zeros_like(raw_probs)  # (n_samples, n_classes)
            n_folds = len(self.iso_maps)
            for fold_maps in self.iso_maps:
                fold_cal = np.zeros_like(raw_probs)
                for cls_idx in range(min(n_classes, len(fold_maps))):
                    m = fold_maps[cls_idx]
                    Xt = m["X_thresholds_"]
                    yt = m["y_thresholds_"]
                    # Extend boundaries so extrapolation is flat
                    xs = np.concatenate([[0.0], Xt, [1.0]])
                    ys = np.concatenate([[yt[0]], yt, [yt[-1]]])
                    fold_cal[:, cls_idx] = np.interp(raw_probs[:, cls_idx], xs, ys)
                cal += fold_cal
            cal /= n_folds
            # Re-normalise rows to sum to 1
            row_sums = cal.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
            return cal / row_sums

        # Path 3: temperature scaling
        logits = np.log(np.clip(raw_probs, 1e-10, 1.0))
        scaled = logits / max(self.temperature, 1e-6)
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def _encode_categorical(self, category: str, encoder_key: str) -> int:
        """Safely encode a categorical value, defaulting to 0 if unknown."""
        mapping = self.encoders.get(encoder_key, {})
        return mapping.get(category, 0)

    def _mahalanobis_distance(self, X_model: np.ndarray) -> Optional[float]:
        """Return the Mahalanobis distance from the training centre,
        or None if OOD stats are unavailable."""
        if not self.ood_stats:
            return None
        from scipy.spatial.distance import mahalanobis
        mean = np.asarray(self.ood_stats["mean"], dtype=np.float64)
        cov_inv = np.asarray(self.ood_stats["cov_inv"], dtype=np.float64)
        x = np.asarray(X_model, dtype=np.float64).ravel()
        if x.shape[0] != mean.shape[0]:
            return None
        return float(mahalanobis(x, mean, cov_inv))

    def _is_out_of_distribution(self, lat: float, lon: float,
                                cal_probs: np.ndarray,
                                X_model: Optional[np.ndarray] = None) -> dict:
        """OOD detection.

        Real farms routinely have irregular soil chemistry (over-fertilised
        plots, salinity, depleted soils) — those are the cases the
        recommendation system exists to help with, not refuse to predict on.
        So feature-space (Mahalanobis) distance is *reported* for diagnostics
        but is no longer treated as a hard OOD verdict; only locations
        outside the trained geographic envelope (Maharashtra) flip ``is_ood``.

        Returns a dict with keys:
            is_ood, geo_outside, mahal_distance, mahal_threshold, reason.
        """
        geo_outside = (lat < MH_LAT_RANGE[0] or lat > MH_LAT_RANGE[1] or
                       lon < MH_LON_RANGE[0] or lon > MH_LON_RANGE[1])
        mahal_d = mahal_thr = None
        if X_model is not None and self.ood_stats is not None:
            mahal_d = self._mahalanobis_distance(X_model)
            mahal_thr = float(self.ood_stats.get("mahal_threshold", 0.0))
        # Mahalanobis is informational only — do NOT contribute to is_ood.
        mahal_ood = False
        low_prob_ood = False
        is_ood = bool(geo_outside)
        reasons = []
        if geo_outside:  reasons.append("location outside Maharashtra")
        if mahal_ood:    reasons.append(
            f"feature-space distance {mahal_d:.2f} > threshold {mahal_thr:.2f}"
        )
        if low_prob_ood: reasons.append(
            f"max calibrated prob {cal_probs.max():.2f} < {OOD_PROB_THRESHOLD}"
        )
        return {
            "is_ood": is_ood,
            "geo_outside": bool(geo_outside),
            "mahal_distance":  mahal_d,
            "mahal_threshold": mahal_thr,
            "reason": "; ".join(reasons),
        }

    def _confidence_flag(self, confidence: float) -> str:
        """Assign confidence tier."""
        if confidence >= HIGH_THRESHOLD:
            return "HIGH"
        elif confidence >= MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _soften_confidence(c: float) -> float:
        """Cap reported confidence below 1.0.

        Real-world recommendation systems never emit a true 100% \u2014 a value
        of exactly 1.0 here is almost always a clamping artifact from
        additive guardrail boosts pushing the calibrated score above 1.
        We squash anything at-or-near 1.0 down to MAX_CONFIDENCE so the
        UI does not lie to the user.
        """
        MAX_CONFIDENCE = 0.97
        if c <= 0.0:
            return 0.0
        return min(MAX_CONFIDENCE, c)

    def _maybe_llm_verify(self, inputs: dict, top3: list, ood_info: dict) -> dict:
        """Run optional LLM second-opinion. Always returns a dict (never raises)."""
        import os
        flag = inputs.get("llm_verify")
        if flag is None:
            env_flag = os.environ.get("CROP_LLM_VERIFY", "auto").lower()
            if env_flag in ("0", "false", "off", "no"):
                return {"enabled": False, "provider": None, "model": None,
                        "verdict": None, "adjusted_top_3": None,
                        "reasoning": None, "concerns": [],
                        "latency_ms": None, "error": "disabled by env"}
        elif not flag:
            return {"enabled": False, "provider": None, "model": None,
                    "verdict": None, "adjusted_top_3": None,
                    "reasoning": None, "concerns": [],
                    "latency_ms": None, "error": "disabled by request"}
        try:
            from models.llm_verifier import verify_recommendation
        except Exception as exc:
            return {"enabled": False, "provider": None, "model": None,
                    "verdict": None, "adjusted_top_3": None,
                    "reasoning": None, "concerns": [],
                    "latency_ms": None, "error": f"import failed: {exc!s}"}
        try:
            return verify_recommendation(inputs, top3, ood_info)
        except Exception as exc:
            return {"enabled": False, "provider": None, "model": None,
                    "verdict": None, "adjusted_top_3": None,
                    "reasoning": None, "concerns": [],
                    "latency_ms": None, "error": f"verify failed: {exc!s}"}

    def _merge_llm_into_top3(self, ml_top3: list, llm_top3: list) -> list:
        """Blend ML and LLM confidences (50/50 on overlapping crops, keep ML metadata).

        New crops introduced by the LLM (replacing one of the ML candidates)
        are appended with the LLM-supplied confidence and a synthetic
        ``llm_only`` marker so downstream consumers can distinguish them.
        """
        ml_by_crop = {c["crop"]: c for c in ml_top3}
        merged = []
        seen = set()
        for entry in llm_top3:
            crop = entry.get("crop")
            if not crop or crop in seen:
                continue
            seen.add(crop)
            llm_conf = float(entry.get("confidence", 0.0))
            base = ml_by_crop.get(crop)
            if base is not None:
                blended = self._soften_confidence(0.5 * float(base["confidence"]) + 0.5 * llm_conf)
                merged.append({**base, "confidence": round(blended, 4),
                               "confidence_pct": f"{blended * 100:.1f}%",
                               "flag": self._confidence_flag(blended),
                               "llm_adjusted": True})
            else:
                soft = self._soften_confidence(llm_conf)
                merged.append({"crop": crop,
                               "confidence": round(soft, 4),
                               "confidence_pct": f"{soft * 100:.1f}%",
                               "flag": self._confidence_flag(soft),
                               "raw_confidence": 0.0,
                               "llm_only": True})
        for c in ml_top3:
            if c["crop"] not in seen and len(merged) < 3:
                merged.append(c)
        merged.sort(key=lambda x: x["confidence"], reverse=True)
        return merged[:3]

    # ──────────────────────────────────────────────────────────────
    # Weather-sensitivity badge (T9)
    # ──────────────────────────────────────────────────────────────

    # Column names in feat_cols that represent the three weather signals.
    # These must match the keys written by _build_feature_vector.
    _WEATHER_PERTURB_COLS = (
        "weather_rainfall_mm",
        "weather_temp_mean",
        "weather_humidity_mean",
    )
    # Corresponding INPUT_BOUNDS keys used to compute ±1σ proxy.
    _WEATHER_BOUND_KEYS = (
        "rainfall",
        "weather_temp",
        "humidity",
    )

    def _weather_sensitivity(self, X_row: np.ndarray) -> dict:
        """Estimate how much the top-1 calibrated confidence shifts under
        ±1σ perturbations of rainfall, air temperature, and humidity.

        σ proxy = (max - min) * 0.10  from INPUT_BOUNDS.

        Returns
        -------
        dict with keys:
            sensitivity_pct : float — max absolute shift in top-1 confidence
                              across all perturbations, expressed as percentage
                              points (0–100).
            label : str — "Low" (<3 pp), "Medium" (3–8 pp), "High" (>8 pp).
        """
        base_probs_raw = self.model.predict_proba(X_row)
        base_cal = self._calibrate(base_probs_raw, X_for_model=X_row)
        base_top1 = float(base_cal[0].max())

        max_delta = 0.0
        for feat_col, bound_key in zip(
                self._WEATHER_PERTURB_COLS, self._WEATHER_BOUND_KEYS):
            # Find column index; skip silently if not present
            if feat_col not in self.feat_cols:
                continue
            col_idx = self.feat_cols.index(feat_col)

            # Compute ±1σ step from INPUT_BOUNDS range × 0.10
            lo, hi, _ = INPUT_BOUNDS.get(bound_key, (0, 1, ""))
            sigma = (hi - lo) * 0.10

            for sign in (+1, -1):
                X_perturb = X_row.copy()
                X_perturb[0, col_idx] = X_perturb[0, col_idx] + sign * sigma
                try:
                    raw_p = self.model.predict_proba(X_perturb)
                    cal_p = self._calibrate(raw_p, X_for_model=X_perturb)
                    top1_p = float(cal_p[0].max())
                    delta = abs(top1_p - base_top1)
                    if delta > max_delta:
                        max_delta = delta
                except Exception:
                    pass  # never crash the main prediction

        sensitivity_pct = round(max_delta * 100, 2)
        if sensitivity_pct < 3.0:
            label = "Low"
        elif sensitivity_pct <= 8.0:
            label = "Medium"
        else:
            label = "High"

        return {"sensitivity_pct": sensitivity_pct, "label": label}

    # Input validation bounds and valid values imported from config module

    def _validate_inputs(self, **kwargs) -> list:
        """Validate input values and return list of warnings/errors.

        Raises ValueError for missing required fields or out-of-range values.
        Returns list of warning strings for soft-boundary issues.
        """
        required = list(INPUT_BOUNDS.keys()) + [
            "soil_type", "soil_texture", "drainage", "agro_zone", "season",
        ]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        warnings_list = []

        # Numeric bounds
        for field, (lo, hi, unit) in INPUT_BOUNDS.items():
            val = kwargs[field]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"'{field}' must be numeric, got {type(val).__name__}: {val}"
                )
            if val < lo or val > hi:
                raise ValueError(
                    f"'{field}' = {val} is out of range [{lo}, {hi}] {unit}"
                )

        # Categorical checks
        season = kwargs.get("season", "")
        if season not in VALID_SEASONS:
            raise ValueError(
                f"'season' must be one of {VALID_SEASONS}, got '{season}'"
            )

        soil = kwargs.get("soil_type", "")
        if soil not in VALID_SOIL_TYPES:
            warnings_list.append(
                f"Unknown soil_type '{soil}' — defaulting encoder to 0. "
                f"Known types: {sorted(VALID_SOIL_TYPES)}"
            )

        # Soft warnings for Maharashtra-specific plausibility
        lat, lon = kwargs.get("lat", 0), kwargs.get("lon", 0)
        if lat < MH_LAT_RANGE[0] or lat > MH_LAT_RANGE[1] or \
           lon < MH_LON_RANGE[0] or lon > MH_LON_RANGE[1]:
            warnings_list.append(
                f"Location ({lat:.2f}, {lon:.2f}) is outside Maharashtra "
                f"training envelope. Prediction may be unreliable."
            )

        # Soil-drainage consistency check
        from generators.crop_profiles import SOIL_DRAINAGE_VALID
        soil = kwargs.get("soil_type", "")
        drainage = kwargs.get("drainage", "")
        if soil in SOIL_DRAINAGE_VALID and drainage:
            valid_drainages = SOIL_DRAINAGE_VALID[soil]
            if drainage not in valid_drainages:
                warnings_list.append(
                    f"⚠️ Inconsistent inputs: '{soil}' soil typically has "
                    f"{valid_drainages} drainage, but '{drainage}' was provided. "
                    f"This combination is physically unlikely and may reduce "
                    f"prediction accuracy."
                )

        # Season-month consistency check
        from generators.crop_profiles import SEASON_MONTHS as SEASON_MONTH_MAP
        month = kwargs.get("month", None)
        if month is not None and season:
            valid_months = SEASON_MONTH_MAP.get(season, [])
            if valid_months and int(month) not in valid_months:
                warnings_list.append(
                    f"⚠️ Season-month mismatch: '{season}' season typically "
                    f"covers months {valid_months}, but month={int(month)} was "
                    f"provided. This may indicate incorrect input."
                )

        # Agro-zone vs location consistency check
        from generators.crop_profiles import AGRO_ZONE_BOUNDS
        agro_zone = kwargs.get("agro_zone", "")
        if agro_zone in AGRO_ZONE_BOUNDS:
            bounds = AGRO_ZONE_BOUNDS[agro_zone]
            lat_lo, lat_hi = bounds["lat"]
            lon_lo, lon_hi = bounds["lon"]
            if lat < lat_lo or lat > lat_hi or lon < lon_lo or lon > lon_hi:
                warnings_list.append(
                    f"⚠️ Zone-location mismatch: coordinates ({lat:.2f}, "
                    f"{lon:.2f}) are outside the typical bounds of "
                    f"'{agro_zone}' (lat {lat_lo}-{lat_hi}, lon "
                    f"{lon_lo}-{lon_hi}). Check agro_zone or coordinates."
                )

        return warnings_list

    # Irrigation multipliers for effective moisture/rainfall
    _IRRIGATION_MULTIPLIERS = {
        "Rainfed":   {"moisture": 1.0, "rainfall": 1.0},
        "Drip":      {"moisture": 1.3, "rainfall": 1.4},   # efficient supplemental
        "Sprinkler": {"moisture": 1.2, "rainfall": 1.3},
        "Flood":     {"moisture": 1.5, "rainfall": 1.6},   # canal/flood irrigation
    }

    def _build_feature_vector(self, **kwargs) -> np.ndarray:
        """Convert raw inputs into a feature vector matching training order."""
        N = kwargs["nitrogen"]
        P = kwargs["phosphorus"]
        K = kwargs["potassium"]
        temp = kwargs["temperature"]
        moist = kwargs["moisture"]
        ec = kwargs["ec"]
        ph = kwargs["ph"]
        wt = kwargs["weather_temp"]
        hum = kwargs["humidity"]
        rain = kwargs["rainfall"]
        sun = kwargs["sunshine"]
        wind = kwargs["wind_speed"]
        lat = kwargs["lat"]
        lon = kwargs["lon"]
        alt = kwargs["altitude"]
        oc = kwargs["organic_carbon"]
        season = kwargs["season"]
        month = kwargs["month"]

        # Adjust moisture/rainfall for irrigation type
        irrigation = kwargs.get("irrigation_type", "Rainfed")
        irr_mult = self._IRRIGATION_MULTIPLIERS.get(irrigation, {"moisture": 1.0, "rainfall": 1.0})
        moist = moist * irr_mult["moisture"]
        rain = rain * irr_mult["rainfall"]

        # Encode categoricals
        dr = self._encode_categorical(kwargs["drainage"], "drainage_class")
        az = self._encode_categorical(kwargs["agro_zone"], "agro_zone")
        # NOTE: soil_type and soil_texture are now one-hot encoded (see base dict below)

        # Soil drainage ordinal (from config.py)
        sdo = SOIL_DRAINAGE_MAP.get(kwargs["soil_type"], 4)

        # Season one-hot (includes Annual — v2026_05)
        kh = 1 if season == "Kharif" else 0
        rb = 1 if season == "Rabi" else 0
        zd = 1 if season == "Zaid" else 0
        an = 1 if season == "Annual" else 0

        # Month cyclical
        ms = math.sin(2 * math.pi * month / 12)
        mc = math.cos(2 * math.pi * month / 12)

        # Build feature dict (npk_ratio_* and altitude_temp_correction removed — redundant)
        base = {
            "sensor_nitrogen": N, "sensor_phosphorus": P, "sensor_potassium": K,
            "sensor_temperature": temp, "sensor_moisture": moist,
            "sensor_ec": ec, "sensor_ph": ph,
            "weather_temp_mean": wt, "weather_humidity_mean": hum,
            "weather_rainfall_mm": rain, "weather_sunshine_hrs": sun,
            "weather_wind_speed": wind,
            "lat": lat, "lon": lon, "altitude_m": alt,
            "organic_carbon_pct": oc,
            # Irrigation signal (0/1 — key for Rice/Sugarcane vs dryland)
            "irrigation_available": kwargs.get("irrigation_available", 0),
            # Engineered
            "moisture_deficit": moist - rain * 0.1,
            "ec_stress_flag": 1 if ec > 4000 else 0,
            "month_sin": ms, "month_cos": mc,
            "soil_drainage_ordinal": sdo,
            # Encoded categoricals (drainage + agro_zone label-encoded)
            "drainage_class_encoded": dr, "agro_zone_encoded": az,
            "is_season_Kharif": kh, "is_season_Rabi": rb, "is_season_Zaid": zd,
            "is_season_Annual": an,
            # Interaction features
            "N_x_P": N * P, "N_x_K": N * K, "P_x_K": P * K,
            "temp_x_moisture": temp * moist,
            "temp_x_humidity": wt * hum,
            "rain_x_humidity": rain * hum,
            "ph_x_ec": ph * ec,
            "moisture_rain_ratio": moist / (rain + 1),
        }

        # One-hot encode soil_type and soil_texture
        soil_type_val = kwargs["soil_type"]
        soil_texture_val = kwargs["soil_texture"]
        base[f"soil_type_{soil_type_val}"] = 1
        base[f"soil_texture_{soil_texture_val}"] = 1
        # All other soil_type_* and soil_texture_* columns default to 0.0
        # via the base.get(c, 0.0) fallback below.

        return np.array([base.get(c, 0.0) for c in self.feat_cols],
                        dtype=np.float64)

    def predict(self, **kwargs) -> dict:
        """Generate top-3 crop recommendation with confidence flags.

        Parameters
        ----------
        nitrogen, phosphorus, potassium : float
            Soil NPK in mg/kg (from sensor)
        temperature : float
            Soil temperature in °C
        moisture : float
            Soil moisture in %RH
        ec : float
            Electrical conductivity in μS/cm
        ph : float
            Soil pH
        weather_temp : float
            Air temperature in °C
        humidity : float
            Relative humidity in %
        rainfall : float
            Seasonal rainfall in mm
        sunshine : float
            Sunshine hours per day
        wind_speed : float
            Wind speed in km/h
        lat, lon : float
            GPS coordinates
        altitude : float
            Elevation in meters
        organic_carbon : float
            Soil organic carbon %
        soil_type : str
            e.g. "Black (Regur)", "Red", "Alluvial", "Laterite", "Sandy"
        soil_texture : str
            e.g. "Clay Loam", "Sandy Loam", "Silty Clay"
        drainage : str
            e.g. "Moderate", "Good", "Poor", "Excessive"
        agro_zone : str
            e.g. "Vidarbha", "Marathwada", "Western Maharashtra"
        season : str
            "Kharif", "Rabi", "Zaid", or "Annual".
            For K/R/Z the top-3 pool also includes perennial (Annual) crops
            such as Mango, Grape, Sugarcane. For "Annual" only perennials
            are returned.
        month : int
            Month number (1-12)
        prev_crop : str, optional
            Name of the crop grown in the previous season. When provided,
            rotation bonuses/penalties are applied to re-rank the top-3.
        irrigation_type : str, optional
            "Rainfed" (default), "Drip", "Sprinkler", or "Flood".
            Adjusts effective moisture/rainfall for crop suitability.

        Returns
        -------
        dict with keys:
            top_3: list of {crop, confidence, season, flag}
            confidence_flag: str (HIGH/MEDIUM/LOW/OUT_OF_DISTRIBUTION)
            advisory: str
            is_ood: bool
        """
        prev_crop = kwargs.pop("prev_crop", None)
        llm_verify_flag = kwargs.pop("llm_verify", None)

        # Validate inputs (raises ValueError on hard errors)
        input_warnings = self._validate_inputs(**kwargs)
        for w in input_warnings:
            logger.warning("Input warning: %s", w)

        X = self._build_feature_vector(**kwargs).reshape(1, -1)
        raw_probs = self.model.predict_proba(X)
        cal_probs = self._calibrate(raw_probs, X_for_model=X)

        # \u2500\u2500 Season-conditional renormalisation \u2500\u2500
        # The model distributes probability across all crop classes; after the
        # season hard-filter only a subset is eligible. Displayed confidence
        # should reflect rank *within that eligible set* (rank-preserving
        # normalization).
        #
        # v2026_05: "Annual" crops (perennials + sugarcane) bypass the seasonal
        # filter — they are candidates regardless of the query season because
        # they grow year-round once established. When the user explicitly
        # requests season="Annual", only annual crops are returned.
        from generators.crop_profiles import CROP_TO_SEASON, ANNUAL_CROP_NAMES
        input_season = kwargs["season"]

        # Iterate by COLUMN index (aligned with cal_probs) — not by encoded
        # int — so pruned classes don't cause out-of-bounds indexing.
        season_mask = np.zeros_like(cal_probs[0], dtype=bool)
        for col_idx, crop_name in self.col_to_crop.items():
            crop_season = CROP_TO_SEASON.get(crop_name, "")
            if input_season == "Annual":
                if crop_season == "Annual":
                    season_mask[col_idx] = True
            else:
                if crop_season == input_season or crop_name in ANNUAL_CROP_NAMES:
                    season_mask[col_idx] = True
        season_mass = float(cal_probs[0][season_mask].sum())

        if season_mass > 1e-9:
            cal_probs_season = cal_probs.copy()
            cal_probs_season[0, ~season_mask] = 0.0
            cal_probs_season[0, season_mask] /= season_mass
        else:
            # Degenerate case: no season-eligible crop has any probability mass.
            cal_probs_season = cal_probs

        # Build season-filtered candidate list (take more than 3 to allow
        # backfilling after guardrails drop zero-confidence entries)
        all_indices = cal_probs_season[0].argsort()[::-1]  # sorted desc
        candidates = []
        for idx in all_indices:
            if not season_mask[idx]:
                continue
            crop_name = self.col_to_crop.get(int(idx), f"Unknown_{int(idx)}")
            conf = float(cal_probs_season[0][idx])
            # Tag each candidate with its true season — for annual crops this
            # will show "Annual" even when the user queried a K/R/Z season.
            crop_season = CROP_TO_SEASON.get(crop_name, input_season)
            candidates.append({
                "crop": crop_name,
                "confidence": round(conf, 4),
                "confidence_pct": f"{conf * 100:.1f}%",
                "flag": self._confidence_flag(conf),
                "season": crop_season,
                "is_annual": crop_name in ANNUAL_CROP_NAMES,
                "raw_confidence": round(float(cal_probs[0][idx]), 4),
            })
            if len(candidates) >= 8:
                break

        # Apply agronomic guardrails (soil/EC/drainage/regional adjustments)
        candidates = self._apply_agronomic_guardrails(candidates, kwargs)

        # Apply rotation bonus/penalty if prev_crop is provided
        if prev_crop is not None:
            candidates = self._apply_rotation_adjustment(candidates, prev_crop)

        top3 = [c for c in candidates if c["confidence"] > 0.001]
        if len(top3) < 1:
            top3 = candidates[:3]
        else:
            top3 = top3[:3]

        # ── Weather-sensitivity badge (T9) ──
        try:
            ws = self._weather_sensitivity(X)
            top3 = [{**entry, "weather_sensitivity": ws} for entry in top3]
        except Exception as _ws_err:
            logger.debug("Weather sensitivity skipped: %s", _ws_err)

        top1_conf = top3[0]["confidence"]
        overall_flag = self._confidence_flag(top1_conf)

        # \u2500\u2500 OOD: Mahalanobis (if stats present) + geographic envelope \u2500\u2500
        ood_info = self._is_out_of_distribution(
            kwargs["lat"], kwargs["lon"], cal_probs[0], X_model=X,
        )
        is_ood = ood_info["is_ood"]
        if is_ood:
            overall_flag = "OUT_OF_DISTRIBUTION"
            logger.warning(
                "OOD at (%s, %s): %s",
                kwargs["lat"], kwargs["lon"], ood_info["reason"],
            )

        # \u2500\u2500 Conformal prediction set (if available) \u2500\u2500
        conformal_set = None
        if self.conformal is not None:
            q = float(self.conformal.get("quantile", 0.0))
            threshold = max(0.0, 1.0 - q)
            idx_sorted = np.argsort(cal_probs_season[0])[::-1]
            conformal_set = [
                self.col_to_crop.get(int(i), f"Unknown_{int(i)}")
                for i in idx_sorted
                if cal_probs_season[0][i] >= threshold
            ]
            # Guarantee non-empty set: always include the top-1
            if not conformal_set and len(idx_sorted) > 0:
                conformal_set = [
                    self.col_to_crop.get(int(idx_sorted[0]),
                                         f"Unknown_{int(idx_sorted[0])}")
                ]

        advisory = self._build_advisory(top3, overall_flag, is_ood, kwargs)
        farmer_advisory = self._build_farmer_advisory(top3, overall_flag, is_ood, kwargs)

        # \u2500\u2500 Optional LLM second-opinion (Groq primary, Gemini fallback) \u2500\u2500
        llm_inputs = {**kwargs, "llm_verify": llm_verify_flag}
        llm_verification = self._maybe_llm_verify(llm_inputs, top3, ood_info)
        if llm_verification.get("enabled") and llm_verification.get("adjusted_top_3"):
            top3 = self._merge_llm_into_top3(top3, llm_verification["adjusted_top_3"])
            top1_conf = top3[0]["confidence"]
            if not is_ood:
                overall_flag = self._confidence_flag(top1_conf)

        return {
            "top_3": top3,
            "confidence_flag": overall_flag,
            "advisory": advisory,
            "farmer_advisory": farmer_advisory,
            "is_ood": is_ood,
            "ood_info": ood_info,
            "conformal_set": conformal_set,
            "conformal_alpha": (self.conformal or {}).get("alpha"),
            "season_mass_raw": season_mass,
            "input_warnings": input_warnings,
            "llm_verification": llm_verification,
        }

    # ──────────────────────────────────────────────────────────────
    # Agronomic guardrails (post-prediction correction)
    # ──────────────────────────────────────────────────────────────

    def _apply_agronomic_guardrails(self, top3: list, inputs: dict) -> list:
        """Apply soil, EC, drainage, and regional guardrails.

        Adjusts confidence scores of top-3 predictions based on agronomic
        hard-rules that the ML model may not have learned from generated data.

        Penalties/bonuses:
          - Soil incompatibility: −15% confidence
          - EC sensitivity breach: −10% confidence (progressive)
          - Drainage incompatibility: −20% for sensitive crops on poor drainage
          - Drainage boost: +10% for tolerant crops on poor drainage
          - Regional dominance boost: +5% for zone staple crops
        """
        from generators.crop_profiles import (
            SOIL_CROP_INCOMPATIBLE, EC_SENSITIVE_CROPS,
            REGIONAL_CROP_DOMINANCE,
            DRAINAGE_TOLERANT_CROPS, DRAINAGE_SENSITIVE_CROPS,
        )

        soil_type = inputs.get("soil_type", "")
        ec = inputs.get("ec", 0)
        agro_zone = inputs.get("agro_zone", "")
        season = inputs.get("season", "")
        drainage = inputs.get("drainage", "")

        incompatible = set(SOIL_CROP_INCOMPATIBLE.get(soil_type, []))
        zone_priors = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(season, {})
        is_poor_drainage = drainage in ("Poor", "Very Poor")

        # When the model collapses to a near-uniform distribution (no signal,
        # e.g. far-OOD soil values from over-fertilised plots) the raw boost
        # logic ties crops at ~13% because every regional staple gets the
        # same +0.08 added to identical noise.  In that case we replace the
        # confidence floor with a weight-derived value from the regional
        # priors so the ranking is meaningful and not symmetric.
        uniform_prob = 1.0 / max(self.n_model_classes, 1)
        top_raw = max(
            (c.get("raw_confidence", 0.0) for c in top3),
            default=0.0,
        )
        no_signal = top_raw < uniform_prob * 1.5  # <= 50% above uniform

        adjusted = []
        for crop_info in top3:
            crop_name = crop_info["crop"]
            adj = 0.0
            notes = []
            base_conf = crop_info["confidence"]

            # When no signal: derive a *base* confidence from the regional
            # dominance weight so ranking is non-degenerate, instead of
            # adding a flat boost on top of uniform noise.
            if no_signal and crop_name in zone_priors:
                weight = zone_priors[crop_name]
                # weight 2.0 -> 0.50, 1.5 -> 0.35, 1.0 -> 0.20 (capped)
                base_conf = max(base_conf, min(0.55, 0.10 + 0.20 * weight))
                notes.append(
                    f"regional prior fallback: {crop_name} (weight {weight:.1f} in {agro_zone} {season})"
                )

            # 1. Soil incompatibility penalty
            if crop_name in incompatible:
                adj -= 0.15
                notes.append(f"soil penalty: {crop_name} unsuited for {soil_type}")

            # 2. EC sensitivity penalty
            if crop_name in EC_SENSITIVE_CROPS:
                threshold = EC_SENSITIVE_CROPS[crop_name]
                if ec > threshold:
                    ec_pen = min(0.10, 0.10 * (ec - threshold) / threshold)
                    adj -= ec_pen
                    notes.append(f"EC penalty: {crop_name} sensitive above {threshold} μS/cm")

            # 3. Drainage-based adjustments (critical for Konkan/waterlogged areas)
            if is_poor_drainage:
                if crop_name in DRAINAGE_TOLERANT_CROPS:
                    adj += 0.15
                    notes.append(
                        f"drainage boost: {crop_name} thrives in {drainage} drainage"
                    )
                elif crop_name in DRAINAGE_SENSITIVE_CROPS:
                    # Scale penalty to reduce confidence significantly
                    adj -= 0.30
                    notes.append(
                        f"drainage penalty: {crop_name} fails in {drainage} drainage "
                        f"(root rot / waterlogging risk)"
                    )

            # 4. Regional dominance boost (only on top of real model signal;
            # in no_signal mode the prior fallback above already supplies a
            # weight-derived base, so we skip the additive boost to avoid
            # double-counting).
            if crop_name in zone_priors and not no_signal:
                weight = zone_priors[crop_name]
                if weight >= 2.0:
                    adj += 0.15  # primary regional staple
                    notes.append(f"regional boost: {crop_name} is a primary {agro_zone} staple")
                elif weight >= 1.5:
                    adj += 0.08  # secondary regional crop
                    notes.append(f"regional boost: {crop_name} is a {agro_zone} staple")

            adj_conf = self._soften_confidence(max(0.0, base_conf + adj))
            adjusted.append({
                **crop_info,
                "confidence": round(adj_conf, 4),
                "confidence_pct": f"{adj_conf * 100:.1f}%",
                "flag": self._confidence_flag(adj_conf),
                "guardrail_notes": notes,
            })

        adjusted.sort(key=lambda x: x["confidence"], reverse=True)
        if no_signal and adjusted:
            for entry in adjusted:
                entry.setdefault("guardrail_notes", []).insert(
                    0,
                    "model uncertain on these inputs — recommendation falls back "
                    "to regional/agronomic priors",
                )
        return adjusted

    def _build_advisory(self, top3: list, flag: str,
                        is_ood: bool, inputs: dict) -> str:
        """Generate human-readable advisory note."""
        crop1 = top3[0]["crop"]
        conf1 = top3[0]["confidence_pct"]

        if is_ood:
            return (
                f"WARNING: Location ({inputs['lat']:.2f}, {inputs['lon']:.2f}) "
                f"is outside the model's training region (Maharashtra). "
                f"Prediction ({crop1} at {conf1}) may be unreliable. "
                f"Consult local agricultural experts."
            )

        if flag == "HIGH":
            return (
                f"Strong recommendation: {crop1} ({conf1} confidence). "
                f"Well-suited for {inputs['soil_type']} soil in "
                f"{inputs['agro_zone']} during {inputs['season']} season."
            )

        if flag == "MEDIUM":
            crop2 = top3[1]["crop"] if len(top3) > 1 else "N/A"
            return (
                f"Moderate confidence: {crop1} ({conf1}). "
                f"Also consider {crop2} ({top3[1]['confidence_pct']}). "
                f"Local soil testing recommended for final decision."
            )

        # LOW
        alternatives = ", ".join(f"{t['crop']} ({t['confidence_pct']})"
                                 for t in top3)
        return (
            f"Low confidence prediction. Top options: {alternatives}. "
            f"Consult your local Krishi Vigyan Kendra (KVK) for "
            f"{inputs['agro_zone']} region-specific advice."
        )

    # ──────────────────────────────────────────────────────────────
    # Structured farmer advisory (for UI panel + mobile API)
    # ──────────────────────────────────────────────────────────────

    def _build_farmer_advisory(self, top3: list, flag: str,
                               is_ood: bool, inputs: dict) -> dict:
        """Generate a structured farmer-friendly advisory dictionary.

        Returns a dict with keys:
            why_this_crop: str — plain-language reason for the recommendation
            warnings: list[str] — actionable warnings from guardrails
            irrigation_tips: str — soil-based irrigation guidance
            next_crop: str — what to plant next season
            sowing_window: str — when to sow/harvest
            soil_health: dict — quick soil parameter assessment
        """
        from generators.crop_profiles import (
            ALL_CROPS, CROP_TO_FAMILY, CROP_TO_SEASON,
            SOIL_PROFILES, REGIONAL_CROP_DOMINANCE,
            EC_SENSITIVE_CROPS, SEASON_MONTHS,
        )

        crop1 = top3[0]["crop"]
        conf1 = top3[0]["confidence_pct"]
        soil_type = inputs.get("soil_type", "")
        agro_zone = inputs.get("agro_zone", "")
        season = inputs.get("season", "")
        ec = inputs.get("ec", 0)
        ph = inputs.get("ph", 7.0)

        # ── Why this crop? ──
        reasons = []
        crop_season = CROP_TO_SEASON.get(crop1, season)
        crop_profile = ALL_CROPS.get(crop_season, {}).get(crop1, {})

        soil_affinity = crop_profile.get("soil_affinity", [])
        if soil_type in soil_affinity:
            reasons.append(f"{crop1} thrives in {soil_type} soil (primary affinity)")
        elif soil_type in crop_profile.get("soil_secondary", []):
            reasons.append(f"{crop1} grows well in {soil_type} soil (secondary fit)")

        zone_crops = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(season, {})
        if crop1 in zone_crops:
            weight = zone_crops[crop1]
            if weight >= 2.0:
                reasons.append(f"It is a primary staple crop of {agro_zone}")
            elif weight >= 1.5:
                reasons.append(f"It is commonly grown in {agro_zone}")

        family = CROP_TO_FAMILY.get(crop1, "Unknown")
        reasons.append(f"Crop family: {family} — suited for {season} season")

        if not reasons:
            reasons.append(f"Model prediction based on soil, weather, and location data")

        why = f"{crop1} ({conf1} confidence). " + ". ".join(reasons) + "."

        # ── Warnings (from guardrail notes) ──
        warnings = []
        for crop_info in top3:
            g_notes = crop_info.get("guardrail_notes", [])
            for note in g_notes:
                crop_name = crop_info["crop"]
                if "penalty" in note:
                    # Convert internal note to farmer-friendly warning
                    if "soil" in note:
                        warnings.append(
                            f"⚠️ {crop_name} may struggle in {soil_type} soil — "
                            f"consider a different crop or soil amendment"
                        )
                    elif "EC" in note:
                        threshold = EC_SENSITIVE_CROPS.get(crop_name, 0)
                        warnings.append(
                            f"⚠️ {crop_name} is sensitive to salinity — your EC "
                            f"({ec} μS/cm) exceeds its tolerance ({threshold} μS/cm). "
                            f"Consider gypsum application or selecting a salt-tolerant crop"
                        )
                    elif "drainage" in note:
                        warnings.append(
                            f"⚠️ {crop_name} needs well-drained soil — "
                            f"improve drainage or choose a waterlogging-tolerant crop"
                        )
                    elif "rotation" in note:
                        warnings.append(f"🔄 {note}")

        if is_ood:
            warnings.insert(0,
                f"🚨 Location is outside Maharashtra training region — "
                f"predictions may be unreliable. Consult local KVK.")

        # ── Irrigation guidance ──
        soil_info = SOIL_PROFILES.get(soil_type, {})
        water_ret = soil_info.get("water_retention", 0.5)
        drainage_rate = soil_info.get("drainage_rate", 0.5)

        if water_ret >= 0.80:
            irr_tip = (f"{soil_type} soil has high water retention ({water_ret:.0%}). "
                       f"Irrigate less frequently — risk of waterlogging. "
                       f"Allow soil to dry between irrigations.")
        elif water_ret >= 0.50:
            irr_tip = (f"{soil_type} soil has moderate water retention ({water_ret:.0%}). "
                       f"Standard irrigation schedule. Monitor moisture during dry spells.")
        else:
            irr_tip = (f"{soil_type} soil has low water retention ({water_ret:.0%}). "
                       f"Irrigate frequently in smaller amounts. "
                       f"Consider mulching to reduce evaporation.")

        # Crop-specific water needs
        rainfall_range = crop_profile.get("rainfall_mm", (0, 0))
        if rainfall_range[1] > 0:
            irr_tip += (f" {crop1} needs {rainfall_range[0]}–{rainfall_range[1]} mm "
                        f"total water during the growing season.")

        # ── Next crop suggestion ──
        preferred_next = self._ROTATION_BONUS.get(family, [])
        if preferred_next:
            next_season_map = {"Kharif": "Rabi", "Rabi": "Zaid", "Zaid": "Kharif"}
            next_season = next_season_map.get(season, "Rabi")
            next_season_crops = ALL_CROPS.get(next_season, {})
            suggestions = []
            for next_crop, profile in next_season_crops.items():
                next_family = CROP_TO_FAMILY.get(next_crop, "")
                if next_family in preferred_next and soil_type in (
                    profile.get("soil_affinity", []) + profile.get("soil_secondary", [])
                ):
                    suggestions.append(next_crop)
            if suggestions:
                next_crop_text = (
                    f"After {crop1} ({family}), plant a {'/'.join(preferred_next)} "
                    f"crop in {next_season}. Good options: {', '.join(suggestions[:3])}"
                )
            else:
                next_crop_text = (
                    f"After {crop1} ({family}), prefer {'/'.join(preferred_next)} "
                    f"family crops in the next season for soil health"
                )
        else:
            next_crop_text = "Rotate with a different crop family next season."

        # ── Sowing window ──
        season_months = SEASON_MONTHS.get(season, [])
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        if season_months:
            sow_start = month_names.get(season_months[0], "?")
            sow_end = month_names.get(season_months[-1], "?")
            current_month = inputs.get("month", 0)
            if current_month in season_months:
                timing = f"✅ Current month is within the {season} window"
            else:
                timing = f"📅 {season} sowing window: {sow_start}–{sow_end}"
            sowing = f"{season} season: {sow_start} to {sow_end}. {timing}"
        else:
            sowing = f"Refer to local agricultural calendar for {season} timing."

        # ── Soil health quick assessment ──
        n = inputs.get("nitrogen", 0)
        p = inputs.get("phosphorus", 0)
        k = inputs.get("potassium", 0)

        def _grade(val, low, high):
            if val < low:
                return "Low"
            elif val > high:
                return "High"
            return "Adequate"

        n_range = crop_profile.get("n_range", (80, 150))
        p_range = crop_profile.get("p_range", (40, 80))
        k_range = crop_profile.get("k_range", (60, 130))

        soil_health = {
            "nitrogen": {"value": n, "status": _grade(n, n_range[0], n_range[1]),
                         "ideal_range": f"{n_range[0]}–{n_range[1]} mg/kg"},
            "phosphorus": {"value": p, "status": _grade(p, p_range[0], p_range[1]),
                           "ideal_range": f"{p_range[0]}–{p_range[1]} mg/kg"},
            "potassium": {"value": k, "status": _grade(k, k_range[0], k_range[1]),
                          "ideal_range": f"{k_range[0]}–{k_range[1]} mg/kg"},
            "ph": {"value": ph, "status": _grade(ph, crop_profile.get("ph_range", (6, 8))[0],
                                                  crop_profile.get("ph_range", (6, 8))[1]),
                   "ideal_range": f"{crop_profile.get('ph_range', (6, 8))[0]}–{crop_profile.get('ph_range', (6, 8))[1]}"},
            "ec": {"value": ec, "status": "Concern" if crop1 in EC_SENSITIVE_CROPS and ec > EC_SENSITIVE_CROPS[crop1] else "OK"},
        }

        return {
            "why_this_crop": why,
            "warnings": warnings,
            "irrigation_tips": irr_tip,
            "next_crop": next_crop_text,
            "sowing_window": sowing,
            "soil_health": soil_health,
        }

    # ──────────────────────────────────────────────────────────────
    # Rotation history adjustment
    # ──────────────────────────────────────────────────────────────

    def _apply_rotation_adjustment(self, top3: list, prev_crop: str) -> list:
        """Re-rank top-3 predictions based on rotation with prev_crop.

        Applies +5% bonus for agronomically beneficial succession and
        -5% penalty for same-family repetition.
        """
        from generators.crop_profiles import CROP_TO_FAMILY

        prev_family = CROP_TO_FAMILY.get(prev_crop, "Unknown")
        preferred = self._ROTATION_BONUS.get(prev_family, [])

        adjusted = []
        for crop_info in top3:
            crop_name = crop_info["crop"]
            family = CROP_TO_FAMILY.get(crop_name, "Unknown")
            bonus = 0.0
            note = ""

            if family in preferred:
                bonus = 0.05
                note = f"rotation bonus: {family} after {prev_family}"
            elif family == prev_family:
                bonus = -0.05
                note = f"rotation penalty: same family ({family}) repeated"

            adj_conf = self._soften_confidence(max(0.0, crop_info["confidence"] + bonus))
            adjusted.append({
                **crop_info,
                "confidence": round(adj_conf, 4),
                "confidence_pct": f"{adj_conf * 100:.1f}%",
                "flag": self._confidence_flag(adj_conf),
                "rotation_note": note,
            })

        adjusted.sort(key=lambda x: x["confidence"], reverse=True)
        return adjusted

    # ──────────────────────────────────────────────────────────────
    # Multi-season rotation planner
    # ──────────────────────────────────────────────────────────────

    # Agronomic rotation preferences: family_just_planted → best_next_families
    _ROTATION_BONUS = {
        "Cereal":    ["Legume", "Oilseed"],      # legume fixes N after cereal
        "Legume":    ["Cereal", "Cash"],           # cereal benefits from N
        "Oilseed":   ["Legume", "Cereal"],
        "Cash":      ["Legume", "Cereal"],
        "Vegetable": ["Legume", "Cereal", "Oilseed"],
    }

    # Typical seasonal weather adjustments for Maharashtra
    _SEASON_WEATHER = {
        "Kharif": {"weather_temp": 28, "humidity": 78, "rainfall": 900,
                   "sunshine": 4.5, "wind_speed": 8, "month": 7,
                   "temperature": 27, "moisture": 60},
        "Rabi":   {"weather_temp": 22, "humidity": 55, "rainfall": 350,
                   "sunshine": 7.0, "wind_speed": 6, "month": 12,
                   "temperature": 20, "moisture": 35},
        "Zaid":   {"weather_temp": 34, "humidity": 40, "rainfall": 120,
                   "sunshine": 9.0, "wind_speed": 10, "month": 4,
                   "temperature": 33, "moisture": 20},
    }

    def plan_rotation(self, **kwargs) -> dict:
        """Generate a full-year crop rotation plan: Kharif → Rabi → Zaid.

        Accepts the same base inputs as ``predict()`` (soil, location, etc.).
        Season-specific weather values are auto-filled from typical Maharashtra
        averages but can be overridden per season via ``season_overrides``.

        Parameters
        ----------
        **kwargs :
            Same inputs as ``predict()`` — soil_type, lat, lon, altitude,
            nitrogen, phosphorus, potassium, ec, ph, organic_carbon,
            soil_texture, drainage, agro_zone.
            Season/weather fields are auto-filled per season.
        season_overrides : dict[str, dict], optional
            Per-season overrides, e.g.
            ``{"Kharif": {"rainfall": 1200}, "Rabi": {"humidity": 45}}``

        Returns
        -------
        dict with keys:
            rotation: list of 3 dicts (one per season) each containing
                      season, recommendation (predict() output), and
                      rotation_note (agronomic rationale).
            summary: str — one-line summary of the plan.
        """
        from generators.crop_profiles import CROP_TO_FAMILY, ALL_CROPS

        season_overrides = kwargs.pop("season_overrides", {})
        season_order = ["Kharif", "Rabi", "Zaid"]
        rotation = []
        prev_family = None

        for season in season_order:
            # Build season-specific inputs
            season_inputs = dict(kwargs)
            season_inputs["season"] = season
            # Apply typical weather defaults
            season_inputs.update(self._SEASON_WEATHER[season])
            # Apply user overrides for this season
            if season in season_overrides:
                season_inputs.update(season_overrides[season])

            # Get prediction
            result = self.predict(**season_inputs)
            top3 = result["top_3"]

            # Apply rotation bonus/penalty to re-rank
            if prev_family is not None:
                preferred = self._ROTATION_BONUS.get(prev_family, [])
                scored = []
                for crop_info in top3:
                    crop_name = crop_info["crop"]
                    family = CROP_TO_FAMILY.get(crop_name, "Unknown")
                    bonus = 0.0
                    note = ""

                    # Bonus for agronomically preferred succession
                    if family in preferred:
                        bonus = 0.05
                        note = f"rotation bonus (+5%): {family} after {prev_family}"
                    # Penalty for same family back-to-back
                    elif family == prev_family:
                        bonus = -0.05
                        note = f"rotation penalty (-5%): same family ({family}) repeated"

                    adjusted_conf = self._soften_confidence(max(0.0, crop_info["confidence"] + bonus))
                    scored.append({
                        **crop_info,
                        "confidence": round(adjusted_conf, 4),
                        "confidence_pct": f"{adjusted_conf * 100:.1f}%",
                        "flag": self._confidence_flag(adjusted_conf),
                        "rotation_note": note,
                    })
                # Re-sort by adjusted confidence
                scored.sort(key=lambda x: x["confidence"], reverse=True)
                top3 = scored
                result["top_3"] = top3

            # Determine family of recommended crop for next iteration
            chosen_crop = top3[0]["crop"]
            prev_family = CROP_TO_FAMILY.get(chosen_crop, "Unknown")

            # Check the recommended crop actually belongs to this season
            season_crops = set(ALL_CROPS.get(season, {}).keys())
            in_season = chosen_crop in season_crops
            season_note = ("" if in_season
                           else f"Note: {chosen_crop} is atypical for {season} season.")

            rotation.append({
                "season": season,
                "recommendation": result,
                "chosen_crop": chosen_crop,
                "crop_family": prev_family,
                "in_season": in_season,
                "season_note": season_note,
            })

        # Build summary
        plan_crops = [r["chosen_crop"] for r in rotation]
        plan_families = [r["crop_family"] for r in rotation]
        summary = (
            f"Rotation plan: {plan_crops[0]} (Kharif, {plan_families[0]}) → "
            f"{plan_crops[1]} (Rabi, {plan_families[1]}) → "
            f"{plan_crops[2]} (Zaid, {plan_families[2]})"
        )

        return {"rotation": rotation, "summary": summary}

    # ──────────────────────────────────────────────────────────────
    # Live weather auto-fetch (Open-Meteo — free, no API key)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def fetch_weather(lat: float, lon: float, timeout: int = 10) -> dict:
        """Fetch current weather from Open-Meteo API for a given location.

        Returns a dict with keys matching ``predict()`` weather params:
            weather_temp, humidity, rainfall, sunshine, wind_speed.

        Uses the *current_weather* + *daily* endpoints for a 7-day average
        of rainfall and sunshine so the values better represent seasonal
        conditions rather than a single-day snapshot.

        Parameters
        ----------
        lat, lon : float
            GPS coordinates.
        timeout : int
            HTTP request timeout in seconds.

        Returns
        -------
        dict  with weather_temp, humidity, rainfall, sunshine, wind_speed.

        Raises
        ------
        RuntimeError  if the API call fails.
        """
        import urllib.request
        import urllib.error

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"precipitation"
            f"&daily=precipitation_sum,sunshine_duration"
            f"&timezone=Asia%2FKolkata&forecast_days=7"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CropRecommender/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                import json as _json
                data = _json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Open-Meteo API request failed: {exc}") from exc

        current = data.get("current", {})
        daily = data.get("daily", {})

        # 7-day average rainfall (mm/day → seasonal proxy)
        precip_daily = daily.get("precipitation_sum", [0])
        avg_rain_per_day = sum(precip_daily) / max(len(precip_daily), 1)
        # Estimate seasonal rainfall (rough 120-day season)
        seasonal_rainfall = round(avg_rain_per_day * 120, 1)

        # 7-day average sunshine (seconds → hours)
        sunshine_secs = daily.get("sunshine_duration", [0])
        avg_sunshine_hrs = round(
            sum(sunshine_secs) / max(len(sunshine_secs), 1) / 3600, 1
        )

        weather = {
            "weather_temp": current.get("temperature_2m", 25.0),
            "humidity": current.get("relative_humidity_2m", 60),
            "rainfall": seasonal_rainfall,
            "sunshine": avg_sunshine_hrs,
            "wind_speed": current.get("wind_speed_10m", 8.0),
        }
        logger.info(
            "Fetched weather for (%.2f, %.2f): temp=%.1f°C hum=%d%% "
            "rain=%.0fmm sun=%.1fh wind=%.1fkm/h",
            lat, lon, weather["weather_temp"], weather["humidity"],
            weather["rainfall"], weather["sunshine"], weather["wind_speed"],
        )
        return weather

    def predict_with_live_weather(self, **kwargs) -> dict:
        """Like ``predict()`` but auto-fills weather fields from Open-Meteo.

        Any explicitly provided weather_temp / humidity / rainfall /
        sunshine / wind_speed values will override the fetched ones.

        Note: Open-Meteo's free tier only exposes a 7-day forecast, so the
        ``rainfall`` value derived from it cannot represent a full-season
        cumulative total (which is what the model was trained on). We always
        substitute the season-typical rainfall from ``_SEASON_WEATHER`` for
        the modeled feature — unless the caller passed an explicit value or
        the live 7-day proxy is already in a sane seasonal range. Same logic
        guards humidity / sunshine when the live fetch returns implausibly
        small numbers (e.g. all-zero forecast windows).
        """
        lat = kwargs["lat"]
        lon = kwargs["lon"]

        try:
            live = self.fetch_weather(lat, lon)
        except RuntimeError as exc:
            logger.warning("Weather fetch failed, using fallback defaults: %s", exc)
            live = self._SEASON_WEATHER.get(kwargs.get("season", "Kharif"), {})

        season_defaults = self._SEASON_WEATHER.get(
            kwargs.get("season", "Kharif"), self._SEASON_WEATHER["Kharif"]
        )

        # Minimum plausible seasonal values; below these we fall back to the
        # season defaults to keep the input on the training manifold.
        _MIN_PLAUSIBLE = {
            "rainfall":  season_defaults.get("rainfall", 400) * 0.40,
            "humidity":  20.0,
            "sunshine":  2.0,
        }

        for key in ("weather_temp", "humidity", "rainfall", "sunshine", "wind_speed"):
            if key in kwargs:
                continue  # explicit override wins
            live_val = live.get(key)
            min_ok = _MIN_PLAUSIBLE.get(key)
            if live_val is None or (min_ok is not None and live_val < min_ok):
                kwargs[key] = season_defaults.get(key, live_val if live_val is not None else 0)
                if key == "rainfall":
                    logger.info(
                        "Live rainfall %.1f mm too low for %s — using season default %.0f mm",
                        live_val or 0.0, kwargs.get("season", "Kharif"), kwargs[key],
                    )
            else:
                kwargs[key] = live_val

        return self.predict(**kwargs)

    # ──────────────────────────────────────────────────────────────
    # Soil amendment / fertilizer gap calculator
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_amendments(
        crop_name: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        field_area_ha: float = 1.0,
    ) -> dict:
        """Calculate NPK fertilizer gap for a target crop.

        Compares current soil NPK (mg/kg) against the crop's ideal midpoint
        from crop_profiles.py and converts the deficit to kg/ha of common
        fertilizers (Urea, DAP, MOP).

        Parameters
        ----------
        crop_name : str
            Target crop, e.g. "Soybean", "Wheat".
        nitrogen, phosphorus, potassium : float
            Current soil NPK in mg/kg.
        field_area_ha : float
            Field area in hectares (default 1.0).

        Returns
        -------
        dict with:
            crop: str,
            current_npk: dict, ideal_npk: dict, gap_npk: dict,
            fertilizer_kg_per_ha: dict (Urea, DAP, MOP),
            total_for_field: dict (scaled to field_area_ha),
            notes: list[str]
        """
        from generators.crop_profiles import ALL_CROPS

        # Find the crop profile
        profile = None
        for _season, crops in ALL_CROPS.items():
            if crop_name in crops:
                profile = crops[crop_name]
                break
        if profile is None:
            return {"error": f"Crop '{crop_name}' not found in profiles."}

        # Ideal midpoints
        ideal_n = (profile["n_range"][0] + profile["n_range"][1]) / 2
        ideal_p = (profile["p_range"][0] + profile["p_range"][1]) / 2
        ideal_k = (profile["k_range"][0] + profile["k_range"][1]) / 2

        # Gaps (only deficit, no excess removal)
        gap_n = max(0, ideal_n - nitrogen)
        gap_p = max(0, ideal_p - phosphorus)
        gap_k = max(0, ideal_k - potassium)

        # Convert mg/kg gap to kg/ha (assuming 2M kg soil per ha in top 20cm)
        # Then to common fertilizers:
        #   Urea = 46% N  →  kg_urea = gap_n_kg / 0.46
        #   DAP  = 46% P2O5 (≈20% P)  →  kg_dap = gap_p_kg / 0.20
        #   MOP  = 60% K2O (≈50% K)   →  kg_mop = gap_k_kg / 0.50
        soil_mass_kg_per_ha = 2_000_000  # top 20 cm
        gap_n_kg = gap_n * soil_mass_kg_per_ha / 1_000_000
        gap_p_kg = gap_p * soil_mass_kg_per_ha / 1_000_000
        gap_k_kg = gap_k * soil_mass_kg_per_ha / 1_000_000

        urea_kg = round(gap_n_kg / 0.46, 1)
        dap_kg = round(gap_p_kg / 0.20, 1)
        mop_kg = round(gap_k_kg / 0.50, 1)

        notes = []
        if gap_n == 0 and gap_p == 0 and gap_k == 0:
            notes.append("Soil NPK meets or exceeds crop requirements — no amendments needed.")
        else:
            if gap_n > 0:
                notes.append(f"Nitrogen deficit: {gap_n:.0f} mg/kg → apply ~{urea_kg} kg Urea/ha")
            if gap_p > 0:
                notes.append(f"Phosphorus deficit: {gap_p:.0f} mg/kg → apply ~{dap_kg} kg DAP/ha")
            if gap_k > 0:
                notes.append(f"Potassium deficit: {gap_k:.0f} mg/kg → apply ~{mop_kg} kg MOP/ha")

        return {
            "crop": crop_name,
            "current_npk": {"N": nitrogen, "P": phosphorus, "K": potassium},
            "ideal_npk": {"N": round(ideal_n, 1), "P": round(ideal_p, 1), "K": round(ideal_k, 1)},
            "gap_npk": {"N": round(gap_n, 1), "P": round(gap_p, 1), "K": round(gap_k, 1)},
            "fertilizer_kg_per_ha": {"Urea": urea_kg, "DAP": dap_kg, "MOP": mop_kg},
            "total_for_field": {
                "Urea_kg": round(urea_kg * field_area_ha, 1),
                "DAP_kg": round(dap_kg * field_area_ha, 1),
                "MOP_kg": round(mop_kg * field_area_ha, 1),
                "field_area_ha": field_area_ha,
            },
            "notes": notes,
        }

    # ──────────────────────────────────────────────────────────────
    # Reverse recommendation (user picks a target crop; engine reports
    # feasibility, deficits, fixes, and yield guidance)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_reverse_recommendation(
        target_crop: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        ph: float,
        ec: float,
        soil_type: str,
        drainage: str,
        rainfall: float,
        weather_temp: float,
        agro_zone: str = "",
        field_area_ha: float = 1.0,
    ) -> dict:
        """Evaluate an explicit user-chosen crop against current field conditions.

        Returns a structured report covering:
            - feasibility (HIGH / MEDIUM / LOW / INFEASIBLE)
            - parameter-wise deficits (NPK, pH, EC, rainfall, temperature)
            - actionable fixes (urea/DAP/MOP, lime/sulphur, irrigation, shading)
            - yield-maximisation tips (agronomy + rotation)

        This is the core of the "Reverse Recommendation" tab: instead of
        asking *which* crop to grow, the farmer says *I want to grow X* and
        the engine reports how far off they are and how to close the gap.
        """
        from generators.crop_profiles import (
            ALL_CROPS, CROP_TO_SEASON, CROP_TO_FAMILY,
            SOIL_CROP_INCOMPATIBLE, EC_SENSITIVE_CROPS, EC_TOLERANT_CROPS,
            DRAINAGE_TOLERANT_CROPS, DRAINAGE_SENSITIVE_CROPS,
            REGIONAL_CROP_DOMINANCE,
        )

        # Locate the crop profile
        profile = None
        crop_season = None
        for s, crops in ALL_CROPS.items():
            if target_crop in crops:
                profile = crops[target_crop]
                crop_season = s
                break
        if profile is None:
            return {"error": f"Crop '{target_crop}' not found in crop profiles."}

        # ── Reuse the amendment helper for NPK gaps & fertilizer math ──
        amend = CropRecommender.calculate_amendments(
            target_crop, nitrogen, phosphorus, potassium, field_area_ha,
        )
        gap_npk = amend["gap_npk"]
        ideal_npk = amend["ideal_npk"]

        # ── pH gap ──
        ph_lo, ph_hi = profile.get("ph_range", (6.0, 7.5))
        ph_fix = None
        if ph < ph_lo:
            ph_delta = round(ph_lo - ph, 2)
            # ~2 t/ha lime raises pH by ~0.5 on most soils
            lime_t = round(ph_delta / 0.5 * 2.0 * field_area_ha, 2)
            ph_fix = (f"Soil is too acidic (pH {ph:.2f} < target {ph_lo}). "
                      f"Apply ~{lime_t} t of agricultural lime for {field_area_ha} ha "
                      f"and re-test after 6 weeks.")
        elif ph > ph_hi:
            ph_delta = round(ph - ph_hi, 2)
            sulphur_kg = round(ph_delta / 0.5 * 500 * field_area_ha, 1)
            ph_fix = (f"Soil is too alkaline (pH {ph:.2f} > target {ph_hi}). "
                      f"Apply ~{sulphur_kg} kg elemental sulphur or gypsum for "
                      f"{field_area_ha} ha; add organic compost to buffer.")

        # ── EC (salinity) gap ──
        ec_limit = EC_SENSITIVE_CROPS.get(target_crop,
                                          EC_TOLERANT_CROPS.get(target_crop, 3000))
        ec_fix = None
        if ec > ec_limit:
            ec_fix = (f"Salinity is high (EC {ec:.0f} > tolerance {ec_limit} μS/cm). "
                      f"Leach salts with 1–2 heavy irrigations on a good-drainage "
                      f"plot, apply gypsum 1 t/ha if sodic, and add organic mulch.")

        # ── Rainfall / irrigation gap ──
        rain_lo, rain_hi = profile.get("rainfall_mm", (400, 1200))
        rain_fix = None
        if rainfall < rain_lo:
            deficit = round(rain_lo - rainfall, 0)
            rain_fix = (f"Rainfall is insufficient ({rainfall:.0f} < {rain_lo} mm). "
                        f"Plan supplemental irrigation of ~{deficit} mm "
                        f"(≈ {deficit * 10 * field_area_ha:.0f} kL for "
                        f"{field_area_ha} ha) across the season.")
        elif rainfall > rain_hi:
            excess = round(rainfall - rain_hi, 0)
            rain_fix = (f"Rainfall is excessive ({rainfall:.0f} > {rain_hi} mm). "
                        f"Improve field drainage (ridges/furrows) to shed ~{excess} mm "
                        f"and avoid waterlogging diseases.")

        # ── Temperature gap ──
        t_lo, t_hi = profile.get("temp_range", (15, 35))
        temp_fix = None
        if weather_temp < t_lo:
            temp_fix = (f"Ambient temperature {weather_temp:.1f}°C is below the "
                        f"crop optimum ({t_lo}–{t_hi}°C). Delay sowing to a warmer "
                        f"window or use poly-mulch to raise soil temperature.")
        elif weather_temp > t_hi:
            temp_fix = (f"Ambient temperature {weather_temp:.1f}°C exceeds the "
                        f"crop optimum ({t_lo}–{t_hi}°C). Shift to early sowing, "
                        f"use shade-nets for seedlings, and irrigate at dawn.")

        # ── Soil-type incompatibility ──
        soil_fix = None
        soil_ok = True
        incompatible = set(SOIL_CROP_INCOMPATIBLE.get(soil_type, []))
        if target_crop in incompatible:
            soil_ok = False
            affinity = profile.get("soil_affinity", [])
            soil_fix = (f"'{target_crop}' is agronomically unsuited for "
                        f"{soil_type} soil. It performs best on "
                        f"{', '.join(affinity) or 'well-drained loamy soils'}. "
                        f"Consider raised beds with imported topsoil, or pick "
                        f"an alternative crop.")

        # ── Drainage check ──
        drainage_fix = None
        is_poor_drainage = drainage in ("Poor", "Very Poor")
        if is_poor_drainage and target_crop in DRAINAGE_SENSITIVE_CROPS:
            drainage_fix = (f"{target_crop} needs well-drained soil but the field "
                            f"has {drainage.lower()} drainage. Build raised beds "
                            f"(15–20 cm) with lateral drains every 6–8 m, or "
                            f"choose a tolerant crop "
                            f"({', '.join(sorted(DRAINAGE_TOLERANT_CROPS))}).")

        # ── Regional fit (positive signal) ──
        regional_note = None
        zone_crops = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(crop_season, {})
        if target_crop in zone_crops:
            w = zone_crops[target_crop]
            if w >= 2.0:
                regional_note = f"{target_crop} is a primary staple of {agro_zone}."
            elif w >= 1.5:
                regional_note = f"{target_crop} is commonly grown in {agro_zone}."

        # ── Aggregate feasibility score ──
        # Start at 100, subtract per blocking issue.
        score = 100
        blockers = []
        if not soil_ok:
            score -= 50; blockers.append("soil_incompatibility")
        if drainage_fix:
            score -= 25; blockers.append("drainage_mismatch")
        if ph_fix:
            score -= 10; blockers.append("pH_out_of_range")
        if ec_fix:
            score -= 10; blockers.append("high_salinity")
        if rain_fix:
            score -= 10; blockers.append("water_mismatch")
        if temp_fix:
            score -= 10; blockers.append("temperature_mismatch")
        if any(v > 0 for v in gap_npk.values()):
            score -= 5; blockers.append("npk_deficit")

        if score >= 85:
            feasibility = "HIGH"
        elif score >= 60:
            feasibility = "MEDIUM"
        elif score >= 30:
            feasibility = "LOW"
        else:
            feasibility = "INFEASIBLE"

        # ── Yield-maximisation tips ──
        family = CROP_TO_FAMILY.get(target_crop, "")
        yield_tips = [
            f"Target plant population: follow state package-of-practices for {target_crop}.",
            "Split-apply nitrogen (25% basal, 50% at tillering, 25% at flowering) "
            "to match crop demand and cut leaching losses.",
            "Apply farmyard manure 5–10 t/ha before sowing to improve soil "
            "structure and micronutrient availability.",
            f"Rotate with a different family next season (current: {family or 'Unknown'}) "
            "— legumes after cereals, cereals after legumes.",
            "Scout weekly for pests; use IPM thresholds before chemical sprays.",
        ]
        if target_crop in ("Rice", "Sugarcane"):
            yield_tips.append("Maintain 2–5 cm standing water through tillering; "
                              "drain 7–10 days before harvest.")
        if target_crop in ("Cotton", "Maize"):
            yield_tips.append("Use drip + fertigation to lift yield 15–20% over "
                              "flood irrigation on clay/black soils.")

        # ── Assemble the fix list (ordered by priority) ──
        fixes = []
        if soil_fix:      fixes.append({"type": "soil",      "action": soil_fix})
        if drainage_fix:  fixes.append({"type": "drainage",  "action": drainage_fix})
        if ph_fix:        fixes.append({"type": "pH",        "action": ph_fix})
        if ec_fix:        fixes.append({"type": "salinity",  "action": ec_fix})
        if rain_fix:      fixes.append({"type": "water",     "action": rain_fix})
        if temp_fix:      fixes.append({"type": "temperature","action": temp_fix})
        for nutr, gap in gap_npk.items():
            if gap > 0:
                fert = {"N": "Urea", "P": "DAP", "K": "MOP"}[nutr]
                kg = amend["fertilizer_kg_per_ha"][fert]
                fixes.append({
                    "type": f"nutrient_{nutr}",
                    "action": (f"{nutr} deficit {gap:.0f} mg/kg — apply "
                               f"~{kg} kg {fert}/ha "
                               f"(~{round(kg * field_area_ha, 1)} kg for "
                               f"{field_area_ha} ha)."),
                })

        return {
            "target_crop": target_crop,
            "crop_season": crop_season,
            "crop_family": family,
            "feasibility": feasibility,
            "feasibility_score": score,
            "blockers": blockers,
            "current": {
                "N": nitrogen, "P": phosphorus, "K": potassium,
                "ph": ph, "ec": ec,
                "soil_type": soil_type, "drainage": drainage,
                "rainfall": rainfall, "temperature": weather_temp,
            },
            "ideal": {
                "N": ideal_npk["N"], "P": ideal_npk["P"], "K": ideal_npk["K"],
                "ph_range": list(profile.get("ph_range", (6.0, 7.5))),
                "ec_limit": ec_limit,
                "rainfall_mm": list(profile.get("rainfall_mm", (400, 1200))),
                "temp_range": list(profile.get("temp_range", (15, 35))),
                "soil_affinity": profile.get("soil_affinity", []),
            },
            "gap_npk": gap_npk,
            "fertilizer_kg_per_ha": amend["fertilizer_kg_per_ha"],
            "total_for_field": amend["total_for_field"],
            "fixes": fixes,
            "yield_tips": yield_tips,
            "regional_note": regional_note,
        }

    # ──────────────────────────────────────────────────────────────
    # T8 - Crop Decision Engine
    # ──────────────────────────────────────────────────────────────
    @classmethod
    def evaluate_crop_decision(cls, target_crop: str, **kwargs) -> dict:
        """
        Evaluate a target crop decision with comprehensive scoring, yield
        estimation, and a 4-phase action plan.
        """
        # Run the basic reverse recommendation to get gaps and fixes
        rev = cls.calculate_reverse_recommendation(
            target_crop=target_crop,
            nitrogen=kwargs.get("nitrogen", 0),
            phosphorus=kwargs.get("phosphorus", 0),
            potassium=kwargs.get("potassium", 0),
            ph=kwargs.get("ph", 7.0),
            ec=kwargs.get("ec", 0),
            soil_type=kwargs.get("soil_type", "Alluvial"),
            drainage=kwargs.get("drainage", "Moderate"),
            rainfall=kwargs.get("rainfall", 0),
            weather_temp=kwargs.get("weather_temp", 25.0),
            agro_zone=kwargs.get("agro_zone", ""),
            field_area_ha=kwargs.get("field_area_ha", 1.0)
        )

        if "error" in rev:
            return rev

        try:
            from generators.crop_profiles import CROP_YIELD_BENCHMARKS, CROP_MSP, CROP_AGRONOMY
        except ImportError:
            CROP_YIELD_BENCHMARKS = {}
            CROP_MSP = {}
            CROP_AGRONOMY = {}

        benchmarks = CROP_YIELD_BENCHMARKS.get(target_crop, {
            "max_t_ha": 5.0, "avg_t_ha": 2.0, "unit": "t/ha", "duration_days": 120
        })
        msp = CROP_MSP.get(target_crop, 20000)
        agronomy = CROP_AGRONOMY.get(target_crop, {
            "sow_months": [6, 7], "harvest_months": [10, 11],
            "seed_rate": "Standard", "spacing": "Standard",
            "irrigation_count": "As required", "irrigation_mm": 50,
            "pest_watch": ["General pests"],
            "fert_splits": ["Basal and Top dressing"], "next_crop": "Legumes"
        })

        # ── Month name helper ──
        _MN = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
        def _mn(months):
            return ", ".join(_MN[m] for m in months if 1 <= m <= 12)

        # ── Calculate Component Scores (out of 100) ──
        gap_npk  = rev.get("gap_npk", {})
        ideal    = rev.get("ideal", {})
        blockers = rev.get("blockers", [])
        fixes    = rev.get("fixes", [])
        fixes_by_type = {f["type"]: f["action"] for f in fixes}

        # NPK score — K weighted 1.5× (K deficits hurt yield/quality most)
        ideal_N = ideal.get("N", 0); ideal_P = ideal.get("P", 0); ideal_K = ideal.get("K", 0)
        gap_N   = gap_npk.get("N", 0); gap_P = gap_npk.get("P", 0); gap_K = gap_npk.get("K", 0)
        npk_w_ideal = ideal_N + ideal_P + ideal_K * 1.5
        npk_w_gap   = gap_N   + gap_P   + gap_K   * 1.5
        npk_score = int(max(0, 100 - (npk_w_gap / max(1, npk_w_ideal)) * 100)) if npk_w_ideal > 0 else 100

        # Soil score — blocker key is "soil_incompatibility"
        soil_score = 40 if "soil_incompatibility" in blockers else 95

        # Water score — graduated by rainfall distance from ideal range
        rain_range  = ideal.get("rainfall_mm", [400, 1200])
        rainfall_val = kwargs.get("rainfall", 0)
        rain_lo, rain_hi = rain_range[0], rain_range[1]
        if "water_mismatch" in blockers:
            if rainfall_val < rain_lo:
                water_score = int(max(30, 100 - ((rain_lo - rainfall_val) / max(1, rain_lo)) * 120))
            else:
                water_score = int(max(30, 100 - ((rainfall_val - rain_hi) / max(1, rain_hi)) * 80))
        else:
            water_score = 90
        if "drainage_mismatch" in blockers:
            water_score = max(20, water_score - 20)

        # Climate score — blocker key is "temperature_mismatch"
        climate_score = 50 if "temperature_mismatch" in blockers else 90

        # Composite: NPK 35%, Water 25%, Soil 20%, Climate 10%, Base 10%
        base_score = rev.get("feasibility_score", 50)
        composite_score = int(
            npk_score   * 0.35 +
            water_score * 0.25 +
            soil_score  * 0.20 +
            climate_score * 0.10 +
            base_score  * 0.10
        )

        # Decision
        if composite_score >= 75:
            decision = "GO"
            label = "✅ Recommended"
        elif composite_score >= 55:
            decision = "CAUTION"
            label = "⚠️ Conditional"
        elif composite_score >= 35:
            decision = "HIGH RISK"
            label = "🔶 Not advised"
        else:
            decision = "NO-GO"
            label = "❌ Not feasible"

        # Yield & Financials — realistic: avg_t_ha is baseline, scales to max at score=100
        area_ha = kwargs.get("field_area_ha", 1.0)
        avg_y   = benchmarks.get("avg_t_ha", benchmarks["max_t_ha"] * 0.5)
        max_y   = benchmarks["max_t_ha"]
        t       = max(0.0, (composite_score - 50) / 50.0)
        est_y   = round(min(max_y, max(avg_y * 0.3, avg_y + (max_y - avg_y) * t)), 2)
        yield_gap    = round(max_y - est_y, 2)
        est_revenue  = int(est_y * msp * area_ha)

        # Action Plan — with month names and specific NPK actions
        gap_parts = []
        if gap_N > 0: gap_parts.append(f"N deficit {gap_N:.0f} mg/kg → {fixes_by_type.get('nutrient_N', 'apply Urea')}")
        if gap_P > 0: gap_parts.append(f"P deficit {gap_P:.0f} mg/kg → {fixes_by_type.get('nutrient_P', 'apply DAP')}")
        if gap_K > 0: gap_parts.append(f"K deficit {gap_K:.0f} mg/kg → {fixes_by_type.get('nutrient_K', 'apply MOP')}")

        fixes_text = str(fixes).lower()
        phase1 = []
        if gap_parts:
            phase1.append("Correct NPK gaps before sowing: " + " | ".join(gap_parts))
        else:
            phase1.append("Soil NPK adequate — no basal amendments needed")
        if "lime" in fixes_text:
            phase1.append(fixes_by_type.get("pH", "Apply agricultural lime to correct acidic pH (re-test in 6 weeks)"))
        if "gypsum" in fixes_text or "sulphur" in fixes_text:
            phase1.append(fixes_by_type.get("salinity", fixes_by_type.get("pH", "Apply gypsum/sulphur to correct EC/pH")))
        if "drainage" in fixes_text:
            phase1.append(fixes_by_type.get("drainage", "Install ridge-furrow or BBF system to improve drainage"))
        if not phase1:
            phase1.append("Field conditions are largely suitable — prepare land as per standard practice")

        phase2 = [
            f"Seed Rate: {agronomy['seed_rate']}",
            f"Spacing: {agronomy['spacing']}",
            f"Optimal Sowing Window: {_mn(agronomy['sow_months'])}",
        ]
        phase3 = [
            f"Irrigation: {agronomy['irrigation_count']} irrigations (~{agronomy['irrigation_mm']} mm each)",
            f"Fertilizer Schedule: {', '.join(agronomy['fert_splits'])}",
            f"Pest Watch: Monitor for {', '.join(agronomy['pest_watch'])}",
        ]
        phase4 = [
            f"Expected Harvest Window: {_mn(agronomy['harvest_months'])}",
            f"Next crop in rotation: {agronomy['next_crop']}",
        ]

        # Risk Factors — rich descriptions with actual mitigation text
        _risk_labels = {
            "soil_incompatibility": f"Soil Incompatibility — {target_crop} is not suited for {kwargs.get('soil_type','')} soil",
            "drainage_mismatch":    f"Waterlogging Risk — poor drainage causes root rot / boll drop in {target_crop}",
            "pH_out_of_range":      f"pH Mismatch — soil pH is outside {target_crop}'s optimal range",
            "high_salinity":        f"Salinity Stress — EC exceeds {target_crop}'s salt tolerance threshold",
            "water_mismatch":       (
                f"Excess Rainfall — {rainfall_val:.0f} mm exceeds {target_crop}'s ideal ({rain_lo}–{rain_hi} mm); "
                f"waterlogging & disease pressure high"
                if rainfall_val > rain_hi else
                f"Rainfall Deficit — {rainfall_val:.0f} mm below {target_crop}'s need ({rain_lo}–{rain_hi} mm); "
                f"supplemental irrigation required"
            ),
            "temperature_mismatch": f"Temperature Stress — air temperature is outside {target_crop}'s optimal growing window",
            "npk_deficit":          (
                f"Nutrient Deficit — soil NPK (N={kwargs.get('nitrogen',0)}, P={kwargs.get('phosphorus',0)}, "
                f"K={kwargs.get('potassium',0)}) vs ideal (N={ideal_N:.0f}, P={ideal_P:.0f}, K={ideal_K:.0f}) mg/kg"
            ),
        }
        _mit_map = {
            "soil_incompatibility": fixes_by_type.get("soil", "Consider an alternative crop or amend soil."),
            "drainage_mismatch":    fixes_by_type.get("drainage", "Install ridge-furrow / BBF system; raise beds 15–20 cm."),
            "pH_out_of_range":      fixes_by_type.get("pH", "Apply lime (acidic soil) or sulphur/gypsum (alkaline); retest pH in 6 weeks."),
            "high_salinity":        fixes_by_type.get("salinity", "Leach salts with 1–2 heavy irrigations; apply gypsum 1 t/ha."),
            "water_mismatch":       fixes_by_type.get("water", "Install ridges/furrows for drainage OR plan supplemental irrigation as needed."),
            "temperature_mismatch": fixes_by_type.get("temperature", "Adjust sowing date; use poly-mulch or shade nets for seedlings."),
            "npk_deficit":          " | ".join(f["action"] for f in fixes if f["type"].startswith("nutrient_"))
                                    or "Apply full NPK as per gap analysis in split doses.",
        }
        risks = [
            {
                "risk": _risk_labels.get(b, b.replace("_", " ").title()),
                "mitigation": _mit_map.get(b, "Review field conditions and apply recommended corrections."),
            }
            for b in blockers
        ]
        if npk_score < 60:
            risks.append({
                "risk": f"Severe Nutrient Deficit — NPK score {npk_score}/100; significant yield loss without correction",
                "mitigation": _mit_map.get("npk_deficit", "Apply full NPK in split doses; consider fertigation for better uptake."),
            })

        return {
            "decision": decision,
            "label": label,
            "composite_score": composite_score,
            "scores": {
                "npk": npk_score,
                "soil": soil_score,
                "water": water_score,
                "climate": climate_score,
                "base_feasibility": base_score
            },
            "yield_estimate": {
                "max_potential": max_y,
                "estimated": est_y,
                "gap": yield_gap,
                "unit": benchmarks["unit"]
            },
            "financials": {
                "msp_per_t": msp,
                "estimated_revenue_inr": est_revenue
            },
            "action_plan": {
                "phase1_pre_sowing": phase1,
                "phase2_sowing": phase2,
                "phase3_growth": phase3,
                "phase4_harvest": phase4
            },
            "risks": risks,
            "base_report": rev
        }

def demo():
    """Run a quick demo with sample inputs."""
    import sys
    import traceback
    outfile = open(BASE_DIR / "inference_demo_results.txt", "w", encoding="utf-8")
    _orig = sys.stdout
    sys.stdout = outfile

    try:
        recommender = CropRecommender()
        print("=" * 80)
        print("CROP RECOMMENDATION ENGINE - Top-3 Inference Demo")
        print("=" * 80)

        test_cases = [
            {"name": "Soybean scenario (Vidarbha Kharif)",
             "args": dict(nitrogen=110, phosphorus=55, potassium=70,
                          temperature=26, moisture=55, ec=1400, ph=6.0,
                          weather_temp=27, humidity=75, rainfall=900,
                          sunshine=4.5, wind_speed=8,
                          lat=20.93, lon=77.75, altitude=343,
                          organic_carbon=0.67,
                          soil_type="Black (Regur)", soil_texture="Clay Loam",
                          drainage="Moderate", agro_zone="Vidarbha",
                          season="Kharif", month=7)},
            {"name": "Onion scenario (Nashik Rabi)",
             "args": dict(nitrogen=125, phosphorus=100, potassium=150,
                          temperature=19, moisture=30, ec=850, ph=6.8,
                          weather_temp=20, humidity=50, rainfall=450,
                          sunshine=7.5, wind_speed=6,
                          lat=20.0, lon=74.0, altitude=580,
                          organic_carbon=0.4,
                          soil_type="Red", soil_texture="Sandy Loam",
                          drainage="Excessive", agro_zone="North Maharashtra",
                          season="Rabi", month=12)},
            {"name": "Out-of-distribution (UP Wheat)",
             "args": dict(nitrogen=150, phosphorus=100, potassium=150,
                          temperature=15, moisture=45, ec=1100, ph=7.0,
                          weather_temp=16, humidity=55, rainfall=400,
                          sunshine=7, wind_speed=5,
                          lat=26.85, lon=80.91, altitude=98,
                          organic_carbon=0.6,
                          soil_type="Alluvial", soil_texture="Silt Loam",
                          drainage="Good", agro_zone="Indo-Gangetic Plain",
                          season="Rabi", month=12)},
        ]

        for tc in test_cases:
            print(f"\n{'-' * 80}")
            print(f"SCENARIO: {tc['name']}")
            print(f"{'-' * 80}")
            result = recommender.predict(**tc["args"])
            print(f"\n  Overall Flag: {result['confidence_flag']}")
            print(f"  OOD:          {result['is_ood']}")
            print(f"\n  Top-3 Recommendations:")
            for i, crop in enumerate(result["top_3"], 1):
                print(f"    {i}. {crop['crop']:20s}  {crop['confidence_pct']:>6s}  [{crop['flag']}]")
            print(f"\n  Advisory: {result['advisory']}")

        print(f"\n{'=' * 80}")
        print("DEMO COMPLETE")
        print(f"{'=' * 80}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = _orig
        outfile.close()
    print(f"Results written to {BASE_DIR / 'inference_demo_results.txt'}")


if __name__ == "__main__":
    demo()
