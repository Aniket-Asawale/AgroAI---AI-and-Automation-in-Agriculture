"""
Preprocessing pipeline for the Crop Recommendation Engine.
Reads crop_recommendation_dataset.csv, applies cleaning, feature engineering,
and encoding, then saves the ML-ready dataset.

Usage: python Crop_Recommendation_Engine/preprocessing.py

Outputs:
  data/processed/features.csv        — numeric features + encoded labels
  data/processed/label_encoders.json  — mappings for categorical encoders
  data/processed/preprocessing_report.txt — summary statistics
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─── Paths ───
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "datasets" / "crop_recommendation_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = PROCESSED_DIR / "features.csv"
OUT_ENCODERS = PROCESSED_DIR / "label_encoders.json"
OUT_REPORT = PROCESSED_DIR / "preprocessing_report.txt"


def load_data() -> pd.DataFrame:
    """Load raw dataset."""
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Data cleaning."""
    initial = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    dupes = initial - len(df)

    # Drop rows with null sensor values
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df = df.dropna(subset=sensor_cols)
    nulls = initial - dupes - len(df)

    print(f"  Cleaned: {dupes} duplicates, {nulls} null sensor rows removed → {len(df)} rows")
    return df.reset_index(drop=True)


def normalize_crop_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1b: Normalize noisy crop labels from legacy LLM corrections to
    canonical names.

    The v2026_05 argmax generator already emits canonical names straight from
    ``crop_profiles.ALL_CROPS``, so the map here only covers legacy synonyms
    that may still appear if older batch CSVs are merged in. It must NEVER
    rewrite a canonical crop to a different crop (e.g. Banana→Sugarcane), as
    that would silently corrupt the target.
    """
    label_map = {
        "Sorghum": "Jowar (Kharif)",
        "Sorghum (Kharif)": "Jowar (Kharif)",
        "Sorghum (Jowar)": "Jowar (Kharif)",
        "Brinjal (Kharif/Rabi)": "Brinjal",
        "Sugarcane (Kharif)": "Sugarcane",
        "Onion (Rabi)": "Onion",
        "Grape (Perennial/Specialized)": "Grape",
        "Lentil (Rabi)": "Lentil",
    }
    before = df["crop_label"].nunique()
    df["crop_label"] = df["crop_label"].replace(label_map)
    after = df["crop_label"].nunique()
    print(f"  Normalized crop labels: {before} → {after} unique classes")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Feature engineering per PLAN.md Section 15."""

    # NOTE: npk_ratio_n/p/k REMOVED — redundant with raw sensor_nitrogen/phosphorus/potassium.
    # Keeping raw values only to reduce feature dilution and improve class separability.

    # Moisture deficit (simplified — no per-crop water need yet)
    df["moisture_deficit"] = df["sensor_moisture"] - df["weather_rainfall_mm"] * 0.1

    # EC stress flag (general threshold: EC > 4000 μS/cm is saline stress)
    df["ec_stress_flag"] = (df["sensor_ec"] > 4000).astype(int)

    # NOTE: altitude_temp_correction REMOVED — highly correlated with weather_temp_mean.
    # Keeping weather_temp_mean (air temp) + sensor_temperature (soil temp) as distinct signals.

    # Season one-hot encoding
    season_dummies = pd.get_dummies(df["season"], prefix="is_season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)

    # Month as cyclical features (sin/cos)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Soil drainage ordinal (Sandy=1 drains fastest → Clay=7 drains slowest, per PLAN.md §15.6)
    drainage_order = {
        "Sandy": 1, "Laterite": 2, "Red": 3, "Alluvial": 4,
        "Medium Black": 5, "Black (Regur)": 6, "Shallow Black": 5, "Clay": 7,
    }
    df["soil_drainage_ordinal"] = df["soil_type"].map(drainage_order).fillna(4)

    print(f"  Engineered {4 + len(season_dummies.columns)} new features")
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Step 3: Encoding for categorical columns.

    - soil_type and soil_texture → One-hot encoding (no false ordinal relationship)
    - drainage_class and agro_zone → LabelEncoder (kept for backward compat)
    - crop_family → LabelEncoder (not used in ML features, only for reference)
    """
    encoders = {}

    # One-hot encode soil_type and soil_texture (fixes false ordinal issue)
    for col in ["soil_type", "soil_texture"]:
        dummies = pd.get_dummies(df[col].fillna("Unknown"), prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        # Store the category names for inference-time reconstruction
        encoders[col] = {"onehot_columns": sorted(dummies.columns.tolist())}

    # Label encode remaining categoricals
    label_encoded_cols = ["drainage_class", "agro_zone", "crop_family"]
    for col in label_encoded_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))
        encoders[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}

    # Target label
    le_target = LabelEncoder()
    df["crop_label_encoded"] = le_target.fit_transform(df["crop_label"])
    encoders["crop_label"] = {label: int(idx) for idx, label in enumerate(le_target.classes_)}

    print(f"  One-hot: soil_type ({len(encoders['soil_type']['onehot_columns'])} cols), "
          f"soil_texture ({len(encoders['soil_texture']['onehot_columns'])} cols)")
    print(f"  LabelEncoded: {len(label_encoded_cols)} + target")
    return df, encoders


def select_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Select only ML-relevant columns for the final feature set."""
    feature_cols = [
        # Sensor readings
        "sensor_nitrogen", "sensor_phosphorus", "sensor_potassium",
        "sensor_temperature", "sensor_moisture", "sensor_ec", "sensor_ph",
        # Weather
        "weather_temp_mean", "weather_humidity_mean", "weather_rainfall_mm",
        "weather_sunshine_hrs", "weather_wind_speed",
        # Location
        "lat", "lon", "altitude_m",
        # Soil numeric
        "organic_carbon_pct",
        # Irrigation (key signal for Rice/Sugarcane vs dryland separation)
        "irrigation_available",
        # Engineered (npk_ratio_* and altitude_temp_correction removed — redundant)
        "moisture_deficit", "ec_stress_flag",
        "month_sin", "month_cos", "soil_drainage_ordinal",
        # Encoded categoricals (drainage + agro_zone remain label-encoded)
        "drainage_class_encoded", "agro_zone_encoded",
        # NOTE: crop_family_encoded intentionally excluded — target leakage
    ]

    # Add one-hot columns for soil_type and soil_texture
    soil_type_cols = [c for c in df.columns if c.startswith("soil_type_")]
    soil_texture_cols = [c for c in df.columns if c.startswith("soil_texture_")]
    feature_cols.extend(sorted(soil_type_cols))
    feature_cols.extend(sorted(soil_texture_cols))

    # Add season dummies (prefixed is_season_ to avoid picking up season_year)
    season_cols = [c for c in df.columns if c.startswith("is_season_")]
    feature_cols.extend(season_cols)

    # Target
    target_col = "crop_label_encoded"

    # Metadata (kept for traceability, not for ML)
    meta_cols = ["location_id", "city", "state", "season_year", "crop_label", "confidence_label", "data_quality_flag"]

    all_cols = meta_cols + feature_cols + [target_col]
    existing = [c for c in all_cols if c in df.columns]

    return df[existing]


def generate_report(df_raw: pd.DataFrame, df_final: pd.DataFrame, encoders: dict) -> str:
    """Generate a summary report of the preprocessing pipeline."""
    lines = [
        "=" * 60,
        "PREPROCESSING REPORT — Crop Recommendation Engine",
        "=" * 60,
        f"\nRaw rows:       {len(df_raw)}",
        f"Final rows:     {len(df_final)}",
        f"Feature cols:   {len([c for c in df_final.columns if c not in ['location_id','city','state','crop_label','confidence_label','data_quality_flag','crop_label_encoded']])}",
        f"Target classes: {len(encoders.get('crop_label', {}))}",
        "\n── Crop Label Mapping ──",
    ]
    for label, idx in sorted(encoders.get("crop_label", {}).items(), key=lambda x: x[1]):
        count = len(df_final[df_final["crop_label"] == label])
        lines.append(f"  {idx:2d} → {label} ({count} rows)")

    lines.append("\n── Feature Statistics ──")
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "crop_label_encoded":
            continue
        lines.append(f"  {col}: min={df_final[col].min():.2f}, max={df_final[col].max():.2f}, mean={df_final[col].mean():.2f}")

    lines.append("\n── Confidence Distribution ──")
    if "confidence_label" in df_final.columns:
        for label, count in df_final["confidence_label"].value_counts().items():
            lines.append(f"  {label}: {count} ({count/len(df_final)*100:.1f}%)")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("PREPROCESSING PIPELINE — Crop Recommendation Engine")
    print("=" * 60)

    # Load
    df = load_data()
    df_raw = df.copy()

    # Step 1: Clean
    print("\n[Step 1] Cleaning...")
    df = clean_data(df)

    # Step 1b: Normalize crop labels
    print("\n[Step 1b] Normalizing crop labels...")
    df = normalize_crop_labels(df)

    # Step 2: Feature Engineering
    print("\n[Step 2] Feature Engineering...")
    df = engineer_features(df)

    # Step 3: Label Encoding
    print("\n[Step 3] Label Encoding...")
    df, encoders = encode_labels(df)

    # Step 4: Select ML Features
    print("\n[Step 4] Selecting ML features...")
    df_final = select_ml_features(df)
    print(f"  Final shape: {df_final.shape}")

    # Save
    df_final.to_csv(OUT_FEATURES, index=False)
    print(f"\n✅ Saved features to {OUT_FEATURES}")

    with open(OUT_ENCODERS, "w", encoding="utf-8") as f:
        json.dump(encoders, f, indent=2)
    print(f"✅ Saved encoders to {OUT_ENCODERS}")

    # Report
    report = generate_report(df_raw, df_final, encoders)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ Saved report to {OUT_REPORT}")

    print(f"\n{report}")


if __name__ == "__main__":
    main()
