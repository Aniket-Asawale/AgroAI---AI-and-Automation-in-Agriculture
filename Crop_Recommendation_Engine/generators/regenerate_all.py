"""
Regenerate ALL batches with realistic seasonal weighting and merge.
Maharashtra-only (v2026_05). Seasons weight 3:2:1:2 (Kharif/Rabi/Zaid/Annual).

Usage: python Crop_Recommendation_Engine/generators/regenerate_all.py
"""

import sys
import glob
import json
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generators.sensor_data_generator import generate_batch
from generators.merge_batches import merge_and_validate
from generators.location_generator import generate_locations_json

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
LOCATIONS_FILE = DATA_DIR / "locations_100.json"

# rows_per_season=7 with weights (3/2/1/2) → 24 rows/season-cluster per year
#   Per location/year: Kharif=21, Rabi=14, Zaid=7, Annual=14  = 56 rows
#   × 3 years × 300 locations ≈ 50,400 rows (target 50k)
ROWS_PER_SEASON_BASE = 7

REGION_BATCHES = [
    ("Vidarbha", "Vidarbha"),
    ("Marathwada", "Marathwada"),
    ("Western Maharashtra", "Western_Maharashtra"),
    ("Konkan", "Konkan"),
    ("North Maharashtra", "North_Maharashtra"),
]


def _purge_stale_batches():
    """Delete any existing batch_*.csv so the merge sees only fresh batches."""
    for stale in glob.glob(str(DATA_DIR / "batch_*.csv")):
        try:
            Path(stale).unlink()
        except OSError:
            pass


def main():
    # 1. Rebuild the expanded (300-location) MH-only seed file on disk
    print("=" * 60)
    print("REBUILDING LOCATIONS FILE (300 Maharashtra-only)")
    print("=" * 60)
    generate_locations_json()

    with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
        all_locs = json.load(f)

    # 2. Purge stale batches so we don't mix old and new data
    _purge_stale_batches()

    # 3. Maharashtra batches by agro_zone
    for zone_name, file_name in REGION_BATCHES:
        locs = [l for l in all_locs if l["agro_zone"] == zone_name]
        if not locs:
            print(f"WARNING: No locations for zone '{zone_name}'")
            continue
        generate_batch(locs, file_name, rows_per_season=ROWS_PER_SEASON_BASE)

    # 4. Merge all + validate
    print("\n" + "=" * 60)
    print("MERGING ALL BATCHES")
    print("=" * 60)
    merge_and_validate()


if __name__ == "__main__":
    main()

