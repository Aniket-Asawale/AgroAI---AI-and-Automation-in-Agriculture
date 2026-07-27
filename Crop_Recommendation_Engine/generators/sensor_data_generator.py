"""
Sensor Data Generator — Produces crop recommendation rows.

Generates sensor + weather + crop label data for any set of locations.
Follows PLAN.md Section 7: NPK depletion curves, moisture-rainfall coupling,
EC seasonal patterns, hard-constraint crop labelling, soft scoring.

Usage:
    from generators.sensor_data_generator import generate_batch
    rows = generate_batch(locations, region_name="Vidarbha")
"""

import json
import random
import math
import csv
from pathlib import Path
from datetime import datetime, timedelta

from generators.crop_profiles import (
    ALL_CROPS, ANNUAL_CROPS, ANNUAL_CROP_NAMES,
    SOIL_PROFILES, SEASON_MONTHS, CROP_TO_FAMILY,
    REGIONAL_CROP_DOMINANCE, EC_SENSITIVE_CROPS, EC_TOLERANT_CROPS,
    DRAINAGE_TOLERANT_CROPS, DRAINAGE_SENSITIVE_CROPS,
)

# ─── Constants ───
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
LOCATIONS_FILE = DATA_DIR / "locations_100.json"

YEARS = [2023, 2024, 2025]
SEASONS = ["Kharif", "Rabi", "Zaid", "Annual"]
# Realistic seasonal weights (Maharashtra State Agriculture Census):
#   Kharif ~38%, Rabi ~25%, Zaid ~12%, Annual/perennial ~25%.
# Annual includes Sugarcane, Banana, Grape, Mango, Pomegranate, Cashew, Coconut.
SEASON_WEIGHTS = {"Kharif": 3, "Rabi": 2, "Zaid": 1, "Annual": 2}
NOISE_SIGMA = 0.02  # 2% Gaussian noise (tighter for better class separation)
ANOMALY_RATE = 0.02  # 2% anomaly injection (reduced for cleaner data)

# Argmax labeling: 5% of rows are deliberately flipped to the 2nd-best candidate
# to inject realistic farmer-choice variance without destroying class separability.
LABEL_FLIP_RATE = 0.05

# Seasonal weather baselines for Maharashtra (lat ~17-21°N)
SEASON_WEATHER = {
    "Kharif": {"temp_mean": (26, 32), "humidity": (65, 90), "rainfall_mm": (80, 350), "sunshine_hrs": (4, 7), "wind_speed": (8, 18)},
    "Rabi":   {"temp_mean": (15, 26), "humidity": (35, 60), "rainfall_mm": (0, 30),   "sunshine_hrs": (7, 10), "wind_speed": (5, 12)},
    "Zaid":   {"temp_mean": (28, 40), "humidity": (20, 45), "rainfall_mm": (0, 20),   "sunshine_hrs": (8, 11), "wind_speed": (8, 15)},
    # Annual crops experience year-round weather — sampled dynamically from an
    # underlying monthly profile in `_generate_sensor_values` so that Banana,
    # Sugarcane, Mango, etc. see the full climate envelope during training.
    "Annual": {"temp_mean": (18, 36), "humidity": (40, 85), "rainfall_mm": (0, 250),  "sunshine_hrs": (6, 10), "wind_speed": (6, 15)},
}

# NPK depletion multipliers through crop cycle
# (pre-sowing, early, mid, late, post-harvest)
NPK_DEPLETION = {
    "N": [1.0, 0.90, 0.78, 0.72, 0.65],
    "P": [1.0, 0.96, 0.92, 0.88, 0.85],
    "K": [1.0, 0.94, 0.87, 0.82, 0.78],
}


def _add_noise(value: float, lo: float, hi: float) -> float:
    """Add Gaussian noise (σ=5% of range) and clamp to [lo, hi]."""
    sigma = NOISE_SIGMA * (hi - lo)
    noisy = value + random.gauss(0, sigma)
    return round(max(lo, min(hi, noisy)), 2)


def _altitude_temp_correction(base_temp: float, altitude_m: int) -> float:
    """Adjust temperature by −0.65°C per 100m altitude."""
    return base_temp - 0.65 * (altitude_m / 100.0)


def _get_soil_profile(soil_type: str) -> dict:
    """Get soil profile, falling back to Alluvial if unknown."""
    return SOIL_PROFILES.get(soil_type, SOIL_PROFILES["Alluvial"])


def _generate_sensor_values(
    soil: dict, season: str, crop_profile: dict, altitude_m: int, crop_stage: int
) -> dict:
    """Generate one row of sensor values based on soil + season + crop requirements."""
    # NPK: start from crop requirement range, apply depletion curve
    n_lo, n_hi = crop_profile["n_range"]
    p_lo, p_hi = crop_profile["p_range"]
    k_lo, k_hi = crop_profile["k_range"]

    n_base = random.uniform(n_lo, n_hi) * NPK_DEPLETION["N"][crop_stage]
    p_base = random.uniform(p_lo, p_hi) * NPK_DEPLETION["P"][crop_stage]
    k_base = random.uniform(k_lo, k_hi) * NPK_DEPLETION["K"][crop_stage]

    # Blend with soil baseline (15% soil influence — crop profile dominates)
    n_val = 0.85 * n_base + 0.15 * soil["n_base"]
    p_val = 0.85 * p_base + 0.15 * soil["p_base"]
    k_val = 0.85 * k_base + 0.15 * soil["k_base"]

    # pH: soil base ± seasonal shift. Annual crops span all seasons → neutral shift.
    ph_shift = {"Kharif": -0.2, "Rabi": 0.1, "Zaid": 0.0, "Annual": 0.0}[season]
    ph_base = soil["ph_base"] + ph_shift
    ph_lo, ph_hi = crop_profile["ph_range"]
    # Bring pH toward crop's preferred range (crop-dominant)
    ph_val = (ph_base * 0.25 + random.uniform(ph_lo, ph_hi) * 0.75)

    # EC: seasonal patterns. Annual = average of all three seasons.
    ec_base = soil["ec_base"]
    ec_mult = {"Kharif": 0.7, "Rabi": 1.3, "Zaid": 1.1, "Annual": 1.0}[season]
    ec_val = ec_base * ec_mult * random.uniform(0.8, 1.2)

    # Weather sampling: for Annual, borrow from a randomly-drawn underlying
    # short season (Kharif/Rabi/Zaid) weighted 3:2:1 so perennials see the
    # full climate envelope instead of a single averaged profile.
    if season == "Annual":
        under_season = random.choices(
            ["Kharif", "Rabi", "Zaid"], weights=[3, 2, 1], k=1,
        )[0]
        sw = SEASON_WEATHER[under_season]
    else:
        sw = SEASON_WEATHER[season]

    # Moisture: coupled with rainfall + soil water retention
    wr = soil["water_retention"]
    rainfall = random.uniform(*sw["rainfall_mm"])
    moisture_base = 15 + wr * 40 + (rainfall / 20.0) * wr
    moisture_val = min(95, max(5, moisture_base * random.uniform(0.85, 1.15)))

    # Temperature: altitude-corrected seasonal
    temp_mean = random.uniform(*sw["temp_mean"])
    temp_val = _altitude_temp_correction(temp_mean, altitude_m)

    # Weather values
    humidity = random.uniform(*sw["humidity"])
    sunshine = random.uniform(*sw["sunshine_hrs"])
    wind = random.uniform(*sw["wind_speed"])

    return {
        "sensor_nitrogen": _add_noise(n_val, 0, 500),
        "sensor_phosphorus": _add_noise(p_val, 0, 300),
        "sensor_potassium": _add_noise(k_val, 0, 500),
        "sensor_temperature": round(_add_noise(temp_val, -5, 55), 1),
        "sensor_moisture": round(_add_noise(moisture_val, 0, 100), 1),
        "sensor_ec": round(_add_noise(ec_val, 50, 5000), 0),
        "sensor_ph": round(_add_noise(ph_val, 3.0, 9.0), 1),
        "weather_temp_mean": round(temp_mean, 1),
        "weather_humidity_mean": round(humidity, 1),
        "weather_rainfall_mm": round(rainfall, 1),
        "weather_sunshine_hrs": round(sunshine, 1),
        "weather_wind_speed": round(wind, 1),
    }


# Boost factor for underrepresented crops — increases their selection probability
# to reduce class imbalance in the generated dataset.
MINORITY_CROP_BOOST = {
    "Rice": 1.4, "Sesame": 1.5, "Black Gram": 1.5,
    "Lentil": 1.3, "Green Gram": 1.3, "Okra": 1.3, "Brinjal": 1.3,
}


def _score_crop(
    crop_name: str, profile: dict, soil_type: str, season: str,
    agro_zone: str = "", ec_value: float = 0.0,
    drainage: str = "", irrigation_available: int = 0,
) -> float:
    """Score a crop's suitability for soil, region, EC, drainage, and irrigation (0-1 scale).

    Factors:
      1. Soil affinity (hard constraint)
      2. Drainage compatibility (critical for waterlogged areas)
      3. Regional crop dominance (agro_zone bias)
      4. EC sensitivity / tolerance
      5. Irrigation availability (key signal for Rice/Sugarcane vs dryland crops)
      6. Minority crop boost (class balance)
    """
    score = 0.0

    # ── 1. Soil affinity (hard gate) ──
    if soil_type in profile.get("soil_affinity", []):
        score += 0.5
    elif soil_type in profile.get("soil_secondary", []):
        score += 0.3
    else:
        return 0.0  # Hard constraint: soil must be at least secondary

    # ── 2. Drainage compatibility (critical feature) ──
    if drainage in ("Poor", "Very Poor"):
        if crop_name in DRAINAGE_TOLERANT_CROPS:
            score *= 2.5  # Strong boost: Rice/Sugarcane thrive in waterlogged
        elif crop_name in DRAINAGE_SENSITIVE_CROPS:
            score *= 0.05  # Near-zero: vegetables/oilseeds die in waterlogging
        else:
            score *= 0.4  # Other crops: reduced but possible
    elif drainage == "Excessive":
        if crop_name in DRAINAGE_TOLERANT_CROPS:
            score *= 0.2  # Rice needs water retention, not excessive drainage

    # ── 3. Regional dominance boost ──
    zone_priors = REGIONAL_CROP_DOMINANCE.get(agro_zone, {}).get(season, {})
    regional_boost = zone_priors.get(crop_name, 1.0)
    score *= regional_boost

    # ── 4. EC sensitivity penalty / tolerance bonus ──
    if ec_value > 0:
        if crop_name in EC_SENSITIVE_CROPS:
            threshold = EC_SENSITIVE_CROPS[crop_name]
            if ec_value > threshold:
                penalty = min(0.8, (ec_value - threshold) / threshold)
                score *= (1.0 - penalty)
        elif crop_name in EC_TOLERANT_CROPS:
            threshold = EC_TOLERANT_CROPS[crop_name]
            if ec_value <= threshold:
                score *= 1.15

    # ── 5. Irrigation availability (critical for Rice/Sugarcane separation) ──
    water_intensive = {"Rice", "Sugarcane"}
    drought_tolerant = {"Soybean", "Pigeonpea (Tur)", "Jowar (Kharif)", "Bajra",
                        "Rabi Jowar", "Safflower", "Chickpea (Gram)", "Linseed"}
    # If regional dominance is high (>=2.0), the region naturally supports the crop
    # (e.g. Konkan monsoon provides water for Rice without irrigation infrastructure)
    regionally_dominant = regional_boost >= 2.0
    if irrigation_available:
        if crop_name in water_intensive:
            score *= 1.8  # Strong boost: irrigated → favour water-heavy crops
        elif crop_name in drought_tolerant:
            score *= 0.7  # Slight penalty: farmer would choose higher-value crops
    else:
        # Rainfed: penalise water-intensive crops UNLESS regionally dominant
        if crop_name in water_intensive:
            if regionally_dominant:
                pass  # No penalty: monsoon/natural water sustains these crops here
            else:
                score *= 0.3  # Harsh penalty: can't grow without water supply
        elif crop_name in drought_tolerant:
            score *= 1.4  # Boost: ideal for rainfed farming

    # ── 6. Tiny jitter for tie-breaking only (not stochastic sampling) ──
    score += random.uniform(0.0, 0.02)

    # ── 7. Boost minority crops ──
    boost = MINORITY_CROP_BOOST.get(crop_name, 1.0)
    score *= boost

    return min(1.0, score)


def _select_crop(
    soil_type: str, season: str,
    agro_zone: str = "", ec_value: float = 0.0,
    drainage: str = "", irrigation_available: int = 0,
) -> tuple[str, dict, float]:
    """Select the best crop for given soil + season + region + drainage + irrigation.

    v2026_05 labeling strategy — ARGMAX with 5% second-best flip:
        * picks the top-scoring crop for the (soil, season, zone, drainage, EC,
          irrigation) context;
        * in 5% of cases flips to the 2nd-best crop to inject realistic
          farmer-choice variance without destroying class separability.
    This replaces the prior top-5 weighted-random scheme which produced
    structural label noise and capped model accuracy at ~80 %.
    """
    season_crops = ALL_CROPS.get(season, {})
    candidates = []

    for crop_name, profile in season_crops.items():
        score = _score_crop(
            crop_name, profile, soil_type, season,
            agro_zone=agro_zone, ec_value=ec_value,
            drainage=drainage, irrigation_available=irrigation_available,
        )
        if score > 0:
            candidates.append((crop_name, profile, score))

    if not candidates:
        # Fallback: pick any crop from this season (with low confidence)
        crop_name = random.choice(list(season_crops.keys()))
        return crop_name, season_crops[crop_name], 0.3

    candidates.sort(key=lambda x: x[2], reverse=True)

    # 5% flip to 2nd-best if it exists and is reasonably competitive (>=70% of top)
    if (len(candidates) >= 2
            and random.random() < LABEL_FLIP_RATE
            and candidates[1][2] >= 0.7 * candidates[0][2]):
        return candidates[1]
    return candidates[0]


def _inject_anomaly(row: dict) -> tuple[dict, str]:
    """Inject realistic anomaly into a row (3-5% of data). Returns (row, anomaly_type)."""
    anomaly_type = random.choice(["sensor_drift", "extreme_weather", "marginal_ph", "high_ec"])

    if anomaly_type == "sensor_drift":
        param = random.choice(["sensor_nitrogen", "sensor_phosphorus", "sensor_potassium"])
        row[param] = round(row[param] * random.uniform(1.2, 1.4), 2)
    elif anomaly_type == "extreme_weather":
        if random.random() > 0.5:  # flood
            row["weather_rainfall_mm"] = round(random.uniform(300, 500), 1)
            row["sensor_moisture"] = round(min(98, row["sensor_moisture"] * 1.5), 1)
        else:  # drought
            row["weather_rainfall_mm"] = 0.0
            row["sensor_moisture"] = round(max(3, row["sensor_moisture"] * 0.3), 1)
    elif anomaly_type == "marginal_ph":
        row["sensor_ph"] = round(max(3.0, min(9.0, row["sensor_ph"] + random.choice([-0.8, 0.8]))), 1)
    elif anomaly_type == "high_ec":
        row["sensor_ec"] = round(row["sensor_ec"] * random.uniform(1.8, 2.5), 0)

    return row, anomaly_type


def _generate_timestamp(year: int, season: str, month: int) -> str:
    """Generate a plausible timestamp within the given month."""
    day = random.randint(1, 28)
    hour = random.randint(6, 18)
    minute = random.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"


def generate_rows_for_location(location: dict, rows_per_season: int = 2) -> list[dict]:
    """Generate all rows for a single location across years and seasons.

    rows_per_season is the BASE count. Actual rows per season are:
      Kharif: base * 3  (~50%)
      Rabi:   base * 2  (~33%)
      Zaid:   base * 1  (~17%)
    """
    soil_type = location["soil_type"]
    soil = _get_soil_profile(soil_type)
    altitude_m = location["altitude_m"]
    agro_zone = location.get("agro_zone", "")
    drainage = location.get("drainage_class", "")
    irrigation_available = location.get("irrigation_available", 0)
    water_source = location.get("water_source", "rainfed")
    rows = []

    # Estimate EC for this location's soil (used for crop selection scoring)
    ec_base = soil["ec_base"]

    for year in YEARS:
        for season in SEASONS:
            months = SEASON_MONTHS[season]
            season_rows = rows_per_season * SEASON_WEIGHTS[season]
            # Seasonal EC estimate for crop selection (Annual uses neutral multiplier)
            ec_mult = {"Kharif": 0.7, "Rabi": 1.3, "Zaid": 1.1, "Annual": 1.0}[season]
            ec_estimate = ec_base * ec_mult

            for _row_idx in range(season_rows):
                # Select crop with regional + EC + drainage + irrigation awareness
                crop_name, crop_profile, confidence = _select_crop(
                    soil_type, season,
                    agro_zone=agro_zone, ec_value=ec_estimate,
                    drainage=drainage, irrigation_available=irrigation_available,
                )

                # Pick a month and crop growth stage
                month = random.choice(months)
                # Adjust year for Rabi Jan/Feb months
                ts_year = year + 1 if season == "Rabi" and month in [1, 2] else year
                crop_stage = random.randint(0, 4)

                # Generate sensor values
                sensor_data = _generate_sensor_values(
                    soil, season, crop_profile, altitude_m, crop_stage
                )

                # Determine confidence label
                if confidence >= 0.6:
                    conf_label = "high"
                elif confidence >= 0.4:
                    conf_label = "medium"
                else:
                    conf_label = "uncertain"

                # Build row
                row = {
                    "location_id": location["location_id"],
                    "city": location["city"],
                    "state": location["state"],
                    "agro_zone": location["agro_zone"],
                    "lat": location["lat"],
                    "lon": location["lon"],
                    "altitude_m": altitude_m,
                    "soil_type": soil_type,
                    "soil_texture": location.get("soil_texture", ""),
                    "drainage_class": location.get("drainage_class", ""),
                    "organic_carbon_pct": location.get("organic_carbon_pct", 0.5),
                    "irrigation_available": irrigation_available,
                    "water_source": water_source,
                    "season": season,
                    "season_year": year,
                    "month": month,
                    "timestamp": _generate_timestamp(ts_year, season, month),
                    **sensor_data,
                    "crop_label": crop_name,
                    "crop_family": CROP_TO_FAMILY.get(crop_name, "Unknown"),
                    "crop_category": season,
                    "confidence_label": conf_label,
                    "data_quality_flag": "clean",
                }

                # Anomaly injection
                if random.random() < ANOMALY_RATE:
                    row, anomaly_type = _inject_anomaly(row)
                    row["data_quality_flag"] = f"anomaly:{anomaly_type}"
                    row["confidence_label"] = "uncertain"

                rows.append(row)

    return rows


def generate_batch(
    locations: list[dict],
    region_name: str,
    rows_per_season: int = 2,
    output_csv: bool = True,
) -> list[dict]:
    """
    Generate sensor data for a batch of locations (one region).

    Args:
        locations: List of location dicts from locations_100.json
        region_name: Name for output file (e.g. "vidarbha")
        rows_per_season: Rows per location per season per year (default 2)
        output_csv: Whether to write a CSV file

    Returns:
        List of all generated row dicts
    """
    all_rows = []
    for loc in locations:
        rows = generate_rows_for_location(loc, rows_per_season)
        all_rows.extend(rows)

    print(f"\n── Batch: {region_name} ──")
    print(f"  Locations: {len(locations)}")
    print(f"  Total rows: {len(all_rows)}")

    # Crop distribution
    crop_counts = {}
    for r in all_rows:
        crop_counts[r["crop_label"]] = crop_counts.get(r["crop_label"], 0) + 1
    print(f"  Unique crops: {len(crop_counts)}")
    for crop, count in sorted(crop_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {crop}: {count}")

    # Season distribution
    season_counts = {}
    for r in all_rows:
        season_counts[r["season"]] = season_counts.get(r["season"], 0) + 1
    print(f"  Season split: {season_counts}")

    # Anomaly count
    anomaly_count = sum(1 for r in all_rows if r["data_quality_flag"] != "clean")
    print(f"  Anomalies: {anomaly_count} ({anomaly_count/len(all_rows)*100:.1f}%)")

    if output_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_file = DATA_DIR / f"batch_{region_name.lower().replace(' ', '_')}.csv"
        fieldnames = list(all_rows[0].keys())

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"  ✅ Saved to {out_file}")

    return all_rows


def load_locations(region_filter: str = None) -> list[dict]:
    """Load locations from JSON, optionally filtering by agro_zone."""
    with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
        locations = json.load(f)

    if region_filter:
        locations = [loc for loc in locations if loc["agro_zone"] == region_filter]

    return locations


if __name__ == "__main__":
    import sys

    # Default: generate Vidarbha batch
    region = sys.argv[1] if len(sys.argv) > 1 else "Vidarbha"
    rows_per = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    locs = load_locations(region)
    if not locs:
        print(f"No locations found for region: {region}")
        sys.exit(1)

    generate_batch(locs, region, rows_per_season=rows_per)
