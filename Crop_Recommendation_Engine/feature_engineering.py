"""
Centralised feature engineering for the Crop Recommendation Engine.

All interaction / derived features are defined HERE and imported by
baseline_models.py, ann_model.py, visualization.py, and inference.py.
This prevents logic drift between training and inference.
"""

import pandas as pd


# Metadata columns excluded from ML features
META_COLS = [
    "location_id", "city", "state", "season_year",
    "crop_label", "confidence_label", "data_quality_flag",
]
TARGET_COL = "crop_label_encoded"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add agronomic interaction features to improve separability.

    Features added (8 total):
        N_x_P, N_x_K, P_x_K          — NPK cross-products
        temp_x_moisture               — soil temp x soil moisture
        temp_x_humidity               — air temp x air humidity
        rain_x_humidity               — rainfall x humidity
        ph_x_ec                       — pH x electrical conductivity
        moisture_rain_ratio           — soil moisture / (rainfall + 1)
    """
    df = df.copy()
    # NPK interactions
    df["N_x_P"] = df["sensor_nitrogen"] * df["sensor_phosphorus"]
    df["N_x_K"] = df["sensor_nitrogen"] * df["sensor_potassium"]
    df["P_x_K"] = df["sensor_phosphorus"] * df["sensor_potassium"]
    # Climate interactions
    df["temp_x_moisture"] = df["sensor_temperature"] * df["sensor_moisture"]
    df["temp_x_humidity"] = df["weather_temp_mean"] * df["weather_humidity_mean"]
    df["rain_x_humidity"] = df["weather_rainfall_mm"] * df["weather_humidity_mean"]
    # Soil-climate
    df["ph_x_ec"] = df["sensor_ph"] * df["sensor_ec"]
    df["moisture_rain_ratio"] = (
        df["sensor_moisture"] / (df["weather_rainfall_mm"] + 1)
    )
    return df


# Raw continuous columns eligible for input-jitter augmentation.
# Binary/flag/ordinal/one-hot columns are deliberately excluded.
CONTINUOUS_JITTER_COLS = [
    "sensor_nitrogen", "sensor_phosphorus", "sensor_potassium",
    "sensor_temperature", "sensor_moisture", "sensor_ec", "sensor_ph",
    "weather_temp_mean", "weather_humidity_mean", "weather_rainfall_mm",
    "weather_sunshine_hrs", "weather_wind_speed",
    "lat", "lon", "altitude_m", "organic_carbon_pct",
    "moisture_deficit",
]

# Prefix-based sets of categorical columns that must never be jittered
# and must be passed to SMOTENC as categorical_features.
CATEGORICAL_PREFIXES = ("soil_type_", "soil_texture_", "is_season_")

# Integer-coded non-prefix categoricals (treated as categorical for SMOTENC).
CATEGORICAL_SCALARS = [
    "ec_stress_flag",            # 0/1
    "irrigation_available",      # 0/1
    "soil_drainage_ordinal",     # 1-7 ordinal
    "drainage_class_encoded",    # label-encoded int
    "agro_zone_encoded",         # label-encoded int
]


def recompute_interactions_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the 8 interaction features from the (possibly jittered)
    raw columns, so physical consistency is preserved after augmentation.
    Operates in-place on the supplied frame and returns it.
    """
    df["N_x_P"] = df["sensor_nitrogen"] * df["sensor_phosphorus"]
    df["N_x_K"] = df["sensor_nitrogen"] * df["sensor_potassium"]
    df["P_x_K"] = df["sensor_phosphorus"] * df["sensor_potassium"]
    df["temp_x_moisture"] = df["sensor_temperature"] * df["sensor_moisture"]
    df["temp_x_humidity"] = df["weather_temp_mean"] * df["weather_humidity_mean"]
    df["rain_x_humidity"] = df["weather_rainfall_mm"] * df["weather_humidity_mean"]
    df["ph_x_ec"] = df["sensor_ph"] * df["sensor_ec"]
    df["moisture_rain_ratio"] = (
        df["sensor_moisture"] / (df["weather_rainfall_mm"] + 1)
    )
    return df


def categorical_indices(feature_cols: list[str]) -> list[int]:
    """Return column indices to pass to SMOTENC as `categorical_features`."""
    out = []
    for i, c in enumerate(feature_cols):
        if c.startswith(CATEGORICAL_PREFIXES) or c in CATEGORICAL_SCALARS:
            out.append(i)
    return out


def continuous_indices(feature_cols: list[str]) -> list[int]:
    """Return column indices of CONTINUOUS_JITTER_COLS present in `feature_cols`."""
    return [i for i, c in enumerate(feature_cols) if c in CONTINUOUS_JITTER_COLS]

