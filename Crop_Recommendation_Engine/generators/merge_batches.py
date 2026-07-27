"""Merge all batch CSVs into unified crop_recommendation_dataset.csv and print stats."""

import csv
import glob
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"


def merge_and_validate():
    batch_files = sorted(glob.glob(str(DATA_DIR / "batch_*.csv")))

    all_rows = []
    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"{os.path.basename(bf)}: {len(rows)} rows")
            all_rows.extend(rows)

    print(f"\nTotal rows: {len(all_rows)}")

    # Write unified CSV
    out = DATA_DIR / "crop_recommendation_dataset.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved unified dataset: {out}")

    # Stats
    states = {}
    crops = {}
    seasons = {}
    quality = {}
    conf_labels = {}
    for r in all_rows:
        states[r["state"]] = states.get(r["state"], 0) + 1
        crops[r["crop_label"]] = crops.get(r["crop_label"], 0) + 1
        seasons[r["season"]] = seasons.get(r["season"], 0) + 1
        quality[r["data_quality_flag"]] = quality.get(r["data_quality_flag"], 0) + 1
        conf_labels[r["confidence_label"]] = conf_labels.get(r["confidence_label"], 0) + 1

    print("\n── State Distribution ──")
    for s, c in sorted(states.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c} ({c/len(all_rows)*100:.1f}%)")

    print(f"\n── Crop Distribution ({len(crops)} crops) ──")
    for cr, c in sorted(crops.items(), key=lambda x: -x[1]):
        print(f"  {cr}: {c}")

    print(f"\n── Season Distribution ──")
    for s, c in sorted(seasons.items()):
        print(f"  {s}: {c}")

    print(f"\n── Data Quality ──")
    for q, c in sorted(quality.items(), key=lambda x: -x[1]):
        print(f"  {q}: {c}")

    print(f"\n── Confidence Labels ──")
    for cl, c in sorted(conf_labels.items(), key=lambda x: -x[1]):
        print(f"  {cl}: {c}")

    mh = sum(c for s, c in states.items() if s == "Maharashtra")
    print(f"\nMaharashtra: {mh}/{len(all_rows)} ({mh/len(all_rows)*100:.1f}%)")

    # ── Validation checks ──
    print("\n══ VALIDATION ══")
    errors = 0

    # Check 1: All sensor values within hardware spec
    for i, r in enumerate(all_rows):
        n = float(r["sensor_nitrogen"])
        p = float(r["sensor_phosphorus"])
        k = float(r["sensor_potassium"])
        ph = float(r["sensor_ph"])
        t = float(r["sensor_temperature"])
        m = float(r["sensor_moisture"])
        ec = float(r["sensor_ec"])

        if not (0 <= n <= 2999):
            print(f"  ❌ Row {i}: N={n} out of range")
            errors += 1
        if not (0 <= p <= 2999):
            print(f"  ❌ Row {i}: P={p} out of range")
            errors += 1
        if not (0 <= k <= 2999):
            print(f"  ❌ Row {i}: K={k} out of range")
            errors += 1
        if not (3.0 <= ph <= 9.0):
            print(f"  ❌ Row {i}: pH={ph} out of range")
            errors += 1
        if not (-40 <= t <= 80):
            print(f"  ❌ Row {i}: Temp={t} out of range")
            errors += 1
        if not (0 <= m <= 100):
            print(f"  ❌ Row {i}: Moisture={m} out of range")
            errors += 1
        if not (0 <= ec <= 20000):
            print(f"  ❌ Row {i}: EC={ec} out of range")
            errors += 1

    # Check 2: No empty crop labels
    empty_crops = sum(1 for r in all_rows if not r["crop_label"].strip())
    if empty_crops:
        print(f"  ❌ {empty_crops} rows with empty crop_label")
        errors += empty_crops

    # Check 3: Season-crop consistency (v2026_05: Annual season bypasses K/R/Z)
    # Pull the authoritative mapping from crop_profiles so we don't drift.
    try:
        from generators.crop_profiles import CROP_TO_SEASON
        season_mismatches = 0
        for r in all_rows:
            expected_season = CROP_TO_SEASON.get(r["crop_label"])
            if (expected_season is not None
                    and expected_season != r["season"]
                    and r["data_quality_flag"] == "clean"):
                season_mismatches += 1
        if season_mismatches:
            print(f"  ⚠️  {season_mismatches} season-crop mismatches (in clean rows)")
    except ImportError:
        pass

    if errors == 0:
        print("  ✅ All sensor values within hardware spec")
        print("  ✅ All crop labels present")
    print(f"\n  Total validation errors: {errors}")
    print(f"  Dataset ready: {'YES' if errors == 0 else 'NO'}")


if __name__ == "__main__":
    merge_and_validate()

