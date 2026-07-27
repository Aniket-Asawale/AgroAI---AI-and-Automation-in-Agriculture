"""
Central configuration for the Crop Recommendation Engine.

All constants, paths, thresholds, and mappings live here.
Import from this module instead of hard-coding values in individual scripts.
"""

from pathlib import Path

# ─── Directory Layout ───
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "datasets" / "crop_recommendation_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
ENCODERS_JSON = PROCESSED_DIR / "label_encoders.json"
REPORT_PATH = PROCESSED_DIR / "model_comparison_report.txt"
PREPROCESSING_REPORT = PROCESSED_DIR / "preprocessing_report.txt"
REGISTRY_DIR = BASE_DIR / "models" / "model_registry"
EVALUATION_DIR = BASE_DIR / "evaluation"

# ─── Model Training ───
MODEL_STAMP = "2026_05"
RANDOM_STATE = 42
# Label noise disabled — calibration (isotonic) handles over-confidence correctly.
# Kept at 0.0 so the _inject_label_noise helper remains available for ablation.
LABEL_NOISE_RATE = 0.0
SMOTE_K_NEIGHBORS = 3
TEMPORAL_TRAIN_YEARS = [2023, 2024]
TEMPORAL_TEST_YEAR = 2025
MIN_CLASS_SAMPLES = 5  # drop classes with fewer samples

# ─── Validation Slice (carved from the last portion of 2024) ───
VAL_FRACTION_OF_2024 = 0.15           # stratified on crop_label

# ─── Grouped-by-location split (prevents geographic leakage) ───
# When True, baseline_models.load_data() splits rows so that all rows from a
# given location_id land in exactly ONE of {train, val, test}. This gives an
# honest generalisation estimate — a location the model has never seen.
USE_GROUP_SPLIT = True
GROUP_TEST_FRAC = 0.15                # ~45 locations → test
GROUP_VAL_FRAC = 0.15                 # ~45 locations → val (45→210 train/val/test of 300)
# ─── Input Jitter Augmentation ───
JITTER_ENABLED = True
JITTER_STD_FRAC = 0.035               # std = frac * (max - min) per continuous column
# ─── Hyperparameter Tuning ───
OPTUNA_TRIALS = 30
OPTUNA_CV_SPLITS = 3
# ─── Calibration ───
CALIBRATION_METHOD = "isotonic"       # {"isotonic", "temperature"}
# ─── Ensembling ───
USE_STACKING = True                   # adopt stacker if val-acc gain >= STACKING_MIN_GAIN
STACKING_MIN_GAIN = 0.005
USE_HIERARCHICAL = True               # benchmark vs flat, keep winner on val macro-F1
# ─── Conformal Prediction ───
CONFORMAL_ALPHA = 0.10                # target 90% marginal coverage
# ─── OOD Detection ───
OOD_MAHAL_PERCENTILE = 99             # flag if test Mahalanobis > 99th train percentile
OOD_DISAGREE_PERCENTILE = 99          # flag if RF per-tree std > 99th train percentile

# ─── Confidence Thresholds ───
CONFIDENCE_HIGH = 0.75
CONFIDENCE_LOW = 0.60
CONFIDENCE_UNCERTAIN = 0.60           # alias for report strings — matches CONFIDENCE_LOW
OOD_PROB_THRESHOLD = 0.30             # legacy fallback (used only if OOD stats file absent)

# ─── Anti-Overfitting Gates (enforced after training) ───
MAX_TRAIN_VAL_GAP = 0.06
MAX_VAL_TEST_GAP = 0.04

# ─── Maharashtra Bounding Box (training envelope) ───
MH_LAT_RANGE = (15.5, 22.5)
MH_LON_RANGE = (72.5, 80.5)

# ─── Soil Drainage Ordinal Mapping ───
# Sandy drains fastest (1) -> Clay drains slowest (7)
SOIL_DRAINAGE_MAP = {
    "Sandy": 1, "Laterite": 2, "Red": 3, "Alluvial": 4,
    "Medium Black": 5, "Black (Regur)": 6, "Shallow Black": 5, "Clay": 7,
}

# ─── Valid Categorical Values ───
# "Annual" covers perennials (Grape, Mango, Pomegranate, Cashew, Coconut,
# Banana) and 12–18-month crops (Sugarcane). Annual crops bypass the hard
# season mask during inference — they can be recommended for any query month.
VALID_SEASONS = {"Kharif", "Rabi", "Zaid", "Annual"}
VALID_SOIL_TYPES = set(SOIL_DRAINAGE_MAP.keys())

# ─── Input Validation Bounds ───
INPUT_BOUNDS = {
    "nitrogen":       (0, 500,   "mg/kg"),
    "phosphorus":     (0, 300,   "mg/kg"),
    "potassium":      (0, 500,   "mg/kg"),
    "temperature":    (0, 55,    "deg C"),
    "moisture":       (0, 100,   "%RH"),
    "ec":             (0, 20000, "uS/cm"),
    "ph":             (3.0, 10.0, ""),
    "weather_temp":   (-5, 55,   "deg C"),
    "humidity":       (0, 100,   "%"),
    "rainfall":       (0, 5000,  "mm"),
    "sunshine":       (0, 14,    "hrs/day"),
    "wind_speed":     (0, 100,   "km/h"),
    "lat":            (5, 40,    "degrees"),
    "lon":            (65, 100,  "degrees"),
    "altitude":       (0, 5000,  "m"),
    "organic_carbon": (0, 10,    "%"),
    "month":          (1, 12,    ""),
}

# ─── Crop Season Mapping (for rotation planner) ───
SEASON_MONTHS = {
    "Kharif": (6, 10),   # June-October
    "Rabi":   (11, 3),   # November-March
    "Zaid":   (3, 6),    # March-June (overlap on boundaries)
    "Annual": (1, 12),   # perennials & long-duration — matches any month
}

