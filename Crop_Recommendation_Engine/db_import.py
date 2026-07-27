"""
Import crop_recommendation_dataset.csv into PostgreSQL (agrosensor DB).
Creates the `crop_recommendation` table and bulk-inserts all rows.

Usage: python Crop_Recommendation_Engine/db_import.py
"""

import csv
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# ─── Config ───
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "proShadow")
DB_NAME = os.getenv("DB_NAME", "agrosensor")

CSV_PATH = Path(__file__).resolve().parent / "data" / "datasets" / "crop_recommendation_dataset.csv"
TABLE_NAME = "crop_recommendation"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    location_id VARCHAR(20) NOT NULL,
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    agro_zone VARCHAR(50) NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    altitude_m INTEGER NOT NULL,
    soil_type VARCHAR(50) NOT NULL,
    soil_texture VARCHAR(50),
    drainage_class VARCHAR(50),
    organic_carbon_pct DOUBLE PRECISION,
    season VARCHAR(10) NOT NULL,
    season_year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    sensor_nitrogen DOUBLE PRECISION,
    sensor_phosphorus DOUBLE PRECISION,
    sensor_potassium DOUBLE PRECISION,
    sensor_temperature DOUBLE PRECISION,
    sensor_moisture DOUBLE PRECISION,
    sensor_ec DOUBLE PRECISION,
    sensor_ph DOUBLE PRECISION,
    weather_temp_mean DOUBLE PRECISION,
    weather_humidity_mean DOUBLE PRECISION,
    weather_rainfall_mm DOUBLE PRECISION,
    weather_sunshine_hrs DOUBLE PRECISION,
    weather_wind_speed DOUBLE PRECISION,
    crop_label VARCHAR(50) NOT NULL,
    crop_family VARCHAR(30),
    crop_category VARCHAR(20),
    confidence_label VARCHAR(20),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cr_location ON {TABLE_NAME}(location_id);
CREATE INDEX IF NOT EXISTS idx_cr_crop ON {TABLE_NAME}(crop_label);
CREATE INDEX IF NOT EXISTS idx_cr_season ON {TABLE_NAME}(season, season_year);
CREATE INDEX IF NOT EXISTS idx_cr_state ON {TABLE_NAME}(state);
CREATE INDEX IF NOT EXISTS idx_cr_confidence ON {TABLE_NAME}(confidence_label);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    location_id, city, state, agro_zone, lat, lon, altitude_m,
    soil_type, soil_texture, drainage_class, organic_carbon_pct,
    season, season_year, month, timestamp,
    sensor_nitrogen, sensor_phosphorus, sensor_potassium,
    sensor_temperature, sensor_moisture, sensor_ec, sensor_ph,
    weather_temp_mean, weather_humidity_mean, weather_rainfall_mm,
    weather_sunshine_hrs, weather_wind_speed,
    crop_label, crop_family, crop_category, confidence_label, data_quality_flag
) VALUES %s
"""

FLOAT_COLS = {
    "lat", "lon", "organic_carbon_pct",
    "sensor_nitrogen", "sensor_phosphorus", "sensor_potassium",
    "sensor_temperature", "sensor_moisture", "sensor_ec", "sensor_ph",
    "weather_temp_mean", "weather_humidity_mean", "weather_rainfall_mm",
    "weather_sunshine_hrs", "weather_wind_speed",
}
INT_COLS = {"altitude_m", "season_year", "month"}


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Run regenerate_all.py first.")
        return

    # Read CSV
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from {CSV_PATH.name}")

    # Convert types
    tuples = []
    cols_order = [
        "location_id", "city", "state", "agro_zone", "lat", "lon", "altitude_m",
        "soil_type", "soil_texture", "drainage_class", "organic_carbon_pct",
        "season", "season_year", "month", "timestamp",
        "sensor_nitrogen", "sensor_phosphorus", "sensor_potassium",
        "sensor_temperature", "sensor_moisture", "sensor_ec", "sensor_ph",
        "weather_temp_mean", "weather_humidity_mean", "weather_rainfall_mm",
        "weather_sunshine_hrs", "weather_wind_speed",
        "crop_label", "crop_family", "crop_category", "confidence_label", "data_quality_flag",
    ]
    for row in rows:
        vals = []
        for col in cols_order:
            v = row[col]
            if col in FLOAT_COLS:
                vals.append(float(v) if v else None)
            elif col in INT_COLS:
                vals.append(int(v) if v else None)
            else:
                vals.append(v if v else None)
        tuples.append(tuple(vals))

    # Connect and import
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME)
    cur = conn.cursor()

    try:
        # Drop and recreate for clean import
        cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME} CASCADE")
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print(f"Created table '{TABLE_NAME}'")

        # Bulk insert in batches of 1000
        batch_size = 1000
        for i in range(0, len(tuples), batch_size):
            batch = tuples[i : i + batch_size]
            execute_values(cur, INSERT_SQL, batch)
        conn.commit()

        # Verify
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cur.fetchone()[0]
        print(f"Imported {count} rows into '{TABLE_NAME}'")

        cur.execute(f"SELECT COUNT(DISTINCT crop_label) FROM {TABLE_NAME}")
        crops = cur.fetchone()[0]
        print(f"Unique crops: {crops}")

        cur.execute(f"SELECT season, COUNT(*) FROM {TABLE_NAME} GROUP BY season ORDER BY COUNT(*) DESC")
        print("Season distribution:")
        for season, cnt in cur.fetchall():
            print(f"  {season}: {cnt}")

        print("\n[PASS] PostgreSQL import complete: {}.{}".format(DB_NAME, TABLE_NAME))

    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()

