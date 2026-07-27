"""
Location Generator — Defines ~300 agricultural locations across Maharashtra.

v2026_05 scope change: the dataset is now Maharashtra-only. The 85 hand-curated
MH seed locations are procedurally expanded to ~300 by perturbing lat/lon and
irrigation mix around each seed, yielding per-sub-zone density of 45–75 points.
Rest-of-India locations are kept in REST_OF_INDIA for optional ablation but are
NOT included in the default ALL_LOCATIONS export.

Generates: data/datasets/locations_100.json

Each location includes:
  - location_id, city, district, state, agro_zone
  - lat, lon, altitude_m
  - soil_type (matches AgroSensor SOIL_PROFILES keys)
  - soil_subtype (Deep/Medium/Shallow for Black soil)
  - soil_texture, drainage_class, organic_carbon_pct
"""

import json
import random
from pathlib import Path

# Output path
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "locations_100.json"

# ─── Location expansion parameters ───
# Target total MH locations after procedural expansion around seeds.
TARGET_TOTAL_LOCATIONS = 300
EXPANSION_SEED = 42
# Per-sub-zone density targets (sum ≈ TARGET_TOTAL_LOCATIONS)
ZONE_TARGETS = {
    "Vidarbha":            75,
    "Marathwada":          60,
    "Western Maharashtra": 65,
    "Konkan":              50,
    "North Maharashtra":   50,
}
# Lat/lon jitter (degrees) applied to seed locations during expansion.
# 0.15° ≈ 16 km — stays within the same taluka/district cluster.
JITTER_LAT = 0.15
JITTER_LON = 0.18
# Altitude jitter (m) — same local terrain band.
JITTER_ALT = 40
# Probability a synthetic location adopts irrigation (independent of seed)
IRRIGATION_PROB_BY_ZONE = {
    "Vidarbha":            0.20,
    "Marathwada":          0.18,
    "Western Maharashtra": 0.62,
    "Konkan":              0.15,
    "North Maharashtra":   0.45,
}
WATER_SOURCES_BY_IRR = {
    "Vidarbha":            ["canal", "borewell", "well"],
    "Marathwada":          ["borewell", "well"],
    "Western Maharashtra": ["canal", "canal", "borewell", "well"],
    "Konkan":              ["well", "tank"],
    "North Maharashtra":   ["canal", "borewell"],
}

# ─── Soil texture & drainage mappings ───
SOIL_PROPERTIES = {
    "Black (Regur)": {
        "Deep":    {"texture": "Clay",       "drainage": "Poor",     "oc_pct": (0.5, 0.9)},
        "Medium":  {"texture": "Clay Loam",  "drainage": "Moderate", "oc_pct": (0.4, 0.7)},
        "Shallow": {"texture": "Silty Clay", "drainage": "Moderate", "oc_pct": (0.3, 0.5)},
    },
    "Red":      {"texture": "Sandy Loam",  "drainage": "Good",       "oc_pct": (0.3, 0.6)},
    "Laterite": {"texture": "Gravelly Loam","drainage": "Excessive",  "oc_pct": (0.2, 0.5)},
    "Alluvial": {"texture": "Silt Loam",   "drainage": "Good",       "oc_pct": (0.5, 1.0)},
    "Sandy":    {"texture": "Sand",        "drainage": "Excessive",   "oc_pct": (0.1, 0.3)},
    "Clay":     {"texture": "Heavy Clay",  "drainage": "Very Poor",   "oc_pct": (0.6, 1.2)},
}


def _soil_props(soil_type: str, subtype: str = "") -> dict:
    """Get texture, drainage, organic_carbon range for a soil type."""
    entry = SOIL_PROPERTIES.get(soil_type, {})
    if isinstance(entry, dict) and subtype and subtype in entry:
        entry = entry[subtype]
    elif isinstance(entry, dict) and any(k in entry for k in ["Deep", "Medium", "Shallow"]):
        entry = entry.get("Medium", entry.get("Deep", {}))
    import random
    oc_lo, oc_hi = entry.get("oc_pct", (0.3, 0.7))
    return {
        "soil_texture": entry.get("texture", "Loam"),
        "drainage_class": entry.get("drainage", "Moderate"),
        "organic_carbon_pct": round(random.uniform(oc_lo, oc_hi), 2),
    }


def _loc(loc_id, city, district, state, zone, lat, lon, alt, soil, subtype="",
         irrigation=0, water_source="rainfed"):
    """Build a single location dict.

    irrigation: 0 = rainfed, 1 = irrigated
    water_source: 'rainfed', 'canal', 'borewell', 'well', 'tank'
    """
    props = _soil_props(soil, subtype)
    return {
        "location_id": loc_id,
        "city": city,
        "district": district,
        "state": state,
        "agro_zone": zone,
        "lat": lat,
        "lon": lon,
        "altitude_m": alt,
        "soil_type": soil,
        "soil_subtype": subtype,
        "irrigation_available": irrigation,
        "water_source": water_source,
        **props,
    }


# ─── VIDARBHA — 25 locations (mostly rainfed; ~20% borewell-irrigated) ───
VIDARBHA = [
    _loc("MH-VID-01", "Nagpur",        "Nagpur",      "Maharashtra", "Vidarbha", 21.1458, 79.0882, 310, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-VID-02", "Amravati",      "Amravati",    "Maharashtra", "Vidarbha", 20.9320, 77.7523, 343, "Black (Regur)", "Medium"),
    _loc("MH-VID-03", "Wardha",        "Wardha",      "Maharashtra", "Vidarbha", 20.7453, 78.6022, 281, "Black (Regur)", "Deep",    irrigation=1, water_source="borewell"),
    _loc("MH-VID-04", "Yavatmal",      "Yavatmal",    "Maharashtra", "Vidarbha", 20.3899, 78.1307, 457, "Black (Regur)", "Medium"),
    _loc("MH-VID-05", "Chandrapur",    "Chandrapur",  "Maharashtra", "Vidarbha", 19.9615, 79.2961, 194, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-VID-06", "Akola",         "Akola",       "Maharashtra", "Vidarbha", 20.7002, 77.0082, 282, "Black (Regur)", "Medium"),
    _loc("MH-VID-07", "Washim",        "Washim",      "Maharashtra", "Vidarbha", 20.1042, 77.1339, 463, "Black (Regur)", "Shallow"),
    _loc("MH-VID-08", "Buldhana",      "Buldhana",    "Maharashtra", "Vidarbha", 20.5293, 76.1842, 654, "Black (Regur)", "Medium"),
    _loc("MH-VID-09", "Gondia",        "Gondia",      "Maharashtra", "Vidarbha", 21.4624, 80.1920, 305, "Red",           ""),
    _loc("MH-VID-10", "Bhandara",      "Bhandara",    "Maharashtra", "Vidarbha", 21.1669, 79.6508, 262, "Red",           "",        irrigation=1, water_source="borewell"),
    _loc("MH-VID-11", "Gadchiroli",    "Gadchiroli",  "Maharashtra", "Vidarbha", 20.1809, 80.0043, 217, "Red",           ""),
    _loc("MH-VID-12", "Hinganghat",    "Wardha",      "Maharashtra", "Vidarbha", 20.5495, 78.8399, 274, "Black (Regur)", "Deep"),
    _loc("MH-VID-13", "Arvi",          "Wardha",      "Maharashtra", "Vidarbha", 20.9900, 78.2283, 304, "Black (Regur)", "Medium"),
    _loc("MH-VID-14", "Darwha",        "Yavatmal",    "Maharashtra", "Vidarbha", 20.3100, 77.7700, 410, "Black (Regur)", "Shallow"),
    _loc("MH-VID-15", "Wani",          "Yavatmal",    "Maharashtra", "Vidarbha", 20.0600, 78.9500, 230, "Black (Regur)", "Medium"),
    _loc("MH-VID-16", "Rajura",        "Chandrapur",  "Maharashtra", "Vidarbha", 19.7800, 79.3700, 213, "Black (Regur)", "Deep"),
    _loc("MH-VID-17", "Morshi",        "Amravati",    "Maharashtra", "Vidarbha", 21.2900, 77.8500, 352, "Black (Regur)", "Medium"),
    _loc("MH-VID-18", "Achalpur",      "Amravati",    "Maharashtra", "Vidarbha", 21.2575, 77.5117, 359, "Black (Regur)", "Medium",  irrigation=1, water_source="borewell"),
    _loc("MH-VID-19", "Telhara",       "Akola",       "Maharashtra", "Vidarbha", 20.8100, 76.8100, 295, "Black (Regur)", "Shallow"),
    _loc("MH-VID-20", "Tumsar",        "Bhandara",    "Maharashtra", "Vidarbha", 21.3800, 79.7400, 277, "Red",           ""),
    _loc("MH-VID-21", "Tirora",        "Gondia",      "Maharashtra", "Vidarbha", 21.3900, 79.9800, 285, "Red",           ""),
    _loc("MH-VID-22", "Katol",         "Nagpur",      "Maharashtra", "Vidarbha", 21.2767, 78.5856, 333, "Black (Regur)", "Deep"),
    _loc("MH-VID-23", "Ramtek",        "Nagpur",      "Maharashtra", "Vidarbha", 21.3950, 79.3283, 342, "Black (Regur)", "Medium"),
    _loc("MH-VID-24", "Umred",         "Nagpur",      "Maharashtra", "Vidarbha", 20.8500, 79.3300, 280, "Black (Regur)", "Deep"),
    _loc("MH-VID-25", "Chimur",        "Chandrapur",  "Maharashtra", "Vidarbha", 20.4500, 79.3400, 250, "Red",           ""),
]

# ─── MARATHWADA — 20 locations (drought-prone; ~15% irrigated via borewell) ───
MARATHWADA = [
    _loc("MH-MTH-01", "Chhatrapati Sambhajinagar", "Chh. Sambhajinagar", "Maharashtra", "Marathwada", 19.8762, 75.3433, 568, "Black (Regur)", "Deep"),
    _loc("MH-MTH-02", "Latur",         "Latur",       "Maharashtra", "Marathwada", 18.3968, 76.5604, 636, "Black (Regur)", "Deep",    irrigation=1, water_source="borewell"),
    _loc("MH-MTH-03", "Nanded",        "Nanded",      "Maharashtra", "Marathwada", 19.1383, 77.3210, 362, "Black (Regur)", "Medium",  irrigation=1, water_source="borewell"),
    _loc("MH-MTH-04", "Dharashiv",     "Dharashiv",   "Maharashtra", "Marathwada", 18.1860, 76.0444, 652, "Black (Regur)", "Shallow"),
    _loc("MH-MTH-05", "Beed",          "Beed",        "Maharashtra", "Marathwada", 18.9891, 75.7601, 524, "Black (Regur)", "Medium"),
    _loc("MH-MTH-06", "Parbhani",      "Parbhani",    "Maharashtra", "Marathwada", 19.2608, 76.7748, 423, "Black (Regur)", "Medium"),
    _loc("MH-MTH-07", "Hingoli",       "Hingoli",     "Maharashtra", "Marathwada", 19.7173, 77.1517, 494, "Black (Regur)", "Shallow"),
    _loc("MH-MTH-08", "Jalna",         "Jalna",       "Maharashtra", "Marathwada", 19.8347, 75.8800, 508, "Black (Regur)", "Medium"),
    _loc("MH-MTH-09", "Ambejogai",     "Beed",        "Maharashtra", "Marathwada", 18.7334, 76.3871, 578, "Black (Regur)", "Deep"),
    _loc("MH-MTH-10", "Udgir",         "Latur",       "Maharashtra", "Marathwada", 18.3920, 77.1140, 514, "Black (Regur)", "Medium"),
    _loc("MH-MTH-11", "Nilanga",       "Latur",       "Maharashtra", "Marathwada", 18.1168, 76.7516, 620, "Black (Regur)", "Shallow"),
    _loc("MH-MTH-12", "Kaij",          "Beed",        "Maharashtra", "Marathwada", 18.8500, 76.2900, 540, "Black (Regur)", "Medium"),
    _loc("MH-MTH-13", "Selu",          "Parbhani",    "Maharashtra", "Marathwada", 19.4500, 76.4400, 450, "Black (Regur)", "Medium"),
    _loc("MH-MTH-14", "Gangakhed",     "Parbhani",    "Maharashtra", "Marathwada", 18.9700, 76.7500, 468, "Black (Regur)", "Deep",    irrigation=1, water_source="well"),
    _loc("MH-MTH-15", "Mukhed",        "Nanded",      "Maharashtra", "Marathwada", 18.9500, 77.1400, 482, "Black (Regur)", "Shallow"),
    _loc("MH-MTH-16", "Aundha Nagnath","Hingoli",     "Maharashtra", "Marathwada", 19.4600, 77.0900, 515, "Black (Regur)", "Medium"),
    _loc("MH-MTH-17", "Partur",        "Jalna",       "Maharashtra", "Marathwada", 19.5900, 76.2300, 538, "Black (Regur)", "Medium"),
    _loc("MH-MTH-18", "Georai",        "Beed",        "Maharashtra", "Marathwada", 19.2600, 75.7300, 502, "Sandy",         ""),
    _loc("MH-MTH-19", "Tuljapur",      "Dharashiv",   "Maharashtra", "Marathwada", 18.0100, 76.0700, 672, "Black (Regur)", "Deep"),
    _loc("MH-MTH-20", "Deglur",        "Nanded",      "Maharashtra", "Marathwada", 18.5200, 77.3800, 425, "Black (Regur)", "Medium"),
]

# ─── WESTERN MAHARASHTRA — 20 locations (heavily irrigated ~60%; Krishna/Bhima canal) ───
WESTERN_MH = [
    _loc("MH-WST-01", "Pune",          "Pune",        "Maharashtra", "Western Maharashtra", 18.5204, 73.8567, 560, "Red",           "",        irrigation=1, water_source="canal"),
    _loc("MH-WST-02", "Nashik",        "Nashik",      "Maharashtra", "Western Maharashtra", 19.9975, 73.7898, 565, "Black (Regur)", "Medium",  irrigation=1, water_source="canal"),
    _loc("MH-WST-03", "Satara",        "Satara",      "Maharashtra", "Western Maharashtra", 17.6805, 74.0183, 745, "Laterite",      "",        irrigation=1, water_source="canal"),
    _loc("MH-WST-04", "Sangli",        "Sangli",      "Maharashtra", "Western Maharashtra", 16.8524, 74.5815, 549, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-WST-05", "Kolhapur",      "Kolhapur",    "Maharashtra", "Western Maharashtra", 16.6950, 74.2333, 569, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-WST-06", "Solapur",       "Solapur",     "Maharashtra", "Western Maharashtra", 17.6599, 75.9064, 458, "Black (Regur)", "Shallow", irrigation=1, water_source="canal"),
    _loc("MH-WST-07", "Ahmednagar",    "Ahmednagar",  "Maharashtra", "Western Maharashtra", 19.0948, 74.7480, 649, "Black (Regur)", "Medium",  irrigation=1, water_source="canal"),
    _loc("MH-WST-08", "Baramati",      "Pune",        "Maharashtra", "Western Maharashtra", 18.1515, 74.5777, 550, "Black (Regur)", "Medium",  irrigation=1, water_source="canal"),
    _loc("MH-WST-09", "Shirur",        "Pune",        "Maharashtra", "Western Maharashtra", 18.8267, 74.3733, 585, "Red",           "",        irrigation=1, water_source="borewell"),
    _loc("MH-WST-10", "Mahabaleshwar", "Satara",      "Maharashtra", "Western Maharashtra", 17.9237, 73.6586, 1353,"Laterite",      ""),
    _loc("MH-WST-11", "Karad",         "Satara",      "Maharashtra", "Western Maharashtra", 17.2862, 74.1832, 583, "Black (Regur)", "Medium",  irrigation=1, water_source="canal"),
    _loc("MH-WST-12", "Miraj",         "Sangli",      "Maharashtra", "Western Maharashtra", 16.8328, 74.6454, 543, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-WST-13", "Vita",          "Sangli",      "Maharashtra", "Western Maharashtra", 17.2719, 74.5383, 610, "Red",           ""),
    _loc("MH-WST-14", "Pandharpur",    "Solapur",     "Maharashtra", "Western Maharashtra", 17.6787, 75.3314, 472, "Black (Regur)", "Shallow", irrigation=1, water_source="canal"),
    _loc("MH-WST-15", "Shrirampur",    "Ahmednagar",  "Maharashtra", "Western Maharashtra", 19.1165, 74.6600, 510, "Black (Regur)", "Medium",  irrigation=1, water_source="well"),
    _loc("MH-WST-16", "Sinnar",        "Nashik",      "Maharashtra", "Western Maharashtra", 19.8444, 73.9950, 595, "Red",           ""),
    _loc("MH-WST-17", "Igatpuri",      "Nashik",      "Maharashtra", "Western Maharashtra", 19.6956, 73.5606, 620, "Laterite",      ""),
    _loc("MH-WST-18", "Nira",          "Pune",        "Maharashtra", "Western Maharashtra", 18.1150, 74.9950, 535, "Alluvial",      "",        irrigation=1, water_source="canal"),
    _loc("MH-WST-19", "Phaltan",       "Satara",      "Maharashtra", "Western Maharashtra", 17.9920, 74.4314, 567, "Black (Regur)", "Medium",  irrigation=1, water_source="canal"),
    _loc("MH-WST-20", "Barshi",        "Solapur",     "Maharashtra", "Western Maharashtra", 18.2325, 75.6913, 490, "Black (Regur)", "Shallow"),
]


# ─── KONKAN — 10 locations (mostly rainfed; heavy monsoon = natural water) ───
KONKAN = [
    _loc("MH-KNK-01", "Mumbai",        "Mumbai",      "Maharashtra", "Konkan", 19.0760, 72.8777, 14,  "Laterite",      ""),
    _loc("MH-KNK-02", "Thane",         "Thane",       "Maharashtra", "Konkan", 19.2183, 72.9783, 7,   "Alluvial",      "",  irrigation=1, water_source="well"),
    _loc("MH-KNK-03", "Ratnagiri",     "Ratnagiri",   "Maharashtra", "Konkan", 16.9902, 73.3120, 11,  "Clay",          ""),
    _loc("MH-KNK-04", "Sindhudurg",    "Sindhudurg",  "Maharashtra", "Konkan", 16.3488, 73.7553, 5,   "Clay",          ""),
    _loc("MH-KNK-05", "Palghar",       "Palghar",     "Maharashtra", "Konkan", 19.6967, 72.7699, 8,   "Alluvial",      ""),
    _loc("MH-KNK-06", "Alibaug",       "Raigad",      "Maharashtra", "Konkan", 18.6488, 72.8748, 6,   "Clay",          ""),
    _loc("MH-KNK-07", "Dapoli",        "Ratnagiri",   "Maharashtra", "Konkan", 17.7550, 73.1821, 250, "Laterite",      ""),
    _loc("MH-KNK-08", "Chiplun",       "Ratnagiri",   "Maharashtra", "Konkan", 17.5320, 73.5050, 18,  "Alluvial",      "",  irrigation=1, water_source="well"),
    _loc("MH-KNK-09", "Kudal",         "Sindhudurg",  "Maharashtra", "Konkan", 16.1400, 73.6900, 42,  "Laterite",      ""),
    _loc("MH-KNK-10", "Panvel",        "Raigad",      "Maharashtra", "Konkan", 18.9894, 73.1175, 12,  "Alluvial",      ""),
]

# ─── NORTH MAHARASHTRA — 10 locations (mixed ~40%; Tapi basin canal + borewell) ───
NORTH_MH = [
    _loc("MH-NTH-01", "Jalgaon",       "Jalgaon",     "Maharashtra", "North Maharashtra", 21.0077, 75.5626, 209, "Black (Regur)", "Deep",    irrigation=1, water_source="canal"),
    _loc("MH-NTH-02", "Dhule",         "Dhule",       "Maharashtra", "North Maharashtra", 20.9042, 74.7749, 345, "Black (Regur)", "Medium",  irrigation=1, water_source="borewell"),
    _loc("MH-NTH-03", "Nandurbar",     "Nandurbar",   "Maharashtra", "North Maharashtra", 21.3691, 74.2406, 207, "Alluvial",      "",        irrigation=1, water_source="canal"),
    _loc("MH-NTH-04", "Chopda",        "Jalgaon",     "Maharashtra", "North Maharashtra", 21.2500, 75.3000, 225, "Black (Regur)", "Medium"),
    _loc("MH-NTH-05", "Amalner",       "Jalgaon",     "Maharashtra", "North Maharashtra", 21.0400, 75.0600, 261, "Black (Regur)", "Medium"),
    _loc("MH-NTH-06", "Shahada",       "Nandurbar",   "Maharashtra", "North Maharashtra", 21.5500, 74.4700, 178, "Alluvial",      "",        irrigation=1, water_source="canal"),
    _loc("MH-NTH-07", "Shirpur",       "Dhule",       "Maharashtra", "North Maharashtra", 21.3500, 74.8800, 275, "Black (Regur)", "Shallow"),
    _loc("MH-NTH-08", "Pachora",       "Jalgaon",     "Maharashtra", "North Maharashtra", 20.6600, 75.3500, 247, "Black (Regur)", "Medium"),
    _loc("MH-NTH-09", "Bhusawal",      "Jalgaon",     "Maharashtra", "North Maharashtra", 21.0440, 75.7840, 215, "Alluvial",      "",        irrigation=1, water_source="canal"),
    _loc("MH-NTH-10", "Sakri",         "Dhule",       "Maharashtra", "North Maharashtra", 20.9800, 74.3100, 380, "Red",           ""),
]

# ─── REST OF INDIA — 15 locations ───
REST_OF_INDIA = [
    _loc("IN-ROI-01", "Delhi",         "New Delhi",   "Delhi",           "Indo-Gangetic Plain", 28.7041, 77.1025, 216, "Alluvial",      ""),
    _loc("IN-ROI-02", "Bangalore",     "Bangalore Urban","Karnataka",    "Deccan Plateau",      12.9716, 77.5946, 920, "Red",           ""),
    _loc("IN-ROI-03", "Hyderabad",     "Hyderabad",   "Telangana",       "Deccan Plateau",      17.3850, 78.4867, 542, "Red",           ""),
    _loc("IN-ROI-04", "Chennai",       "Chennai",     "Tamil Nadu",      "Coastal Plain",       13.0827, 80.2707, 6,   "Sandy",         ""),
    _loc("IN-ROI-05", "Jaipur",        "Jaipur",      "Rajasthan",       "Arid Western",        26.9124, 75.7873, 431, "Sandy",         ""),
    _loc("IN-ROI-06", "Lucknow",       "Lucknow",     "Uttar Pradesh",   "Indo-Gangetic Plain", 26.8467, 80.9462, 123, "Alluvial",      ""),
    _loc("IN-ROI-07", "Ahmedabad",     "Ahmedabad",   "Gujarat",         "Semi-Arid Gujarat",   23.0225, 72.5714, 53,  "Alluvial",      ""),
    _loc("IN-ROI-08", "Indore",        "Indore",      "Madhya Pradesh",  "Central Plateau",     22.7196, 75.8577, 553, "Black (Regur)", "Deep"),
    _loc("IN-ROI-09", "Bhopal",        "Bhopal",      "Madhya Pradesh",  "Central Plateau",     23.2599, 77.4126, 527, "Black (Regur)", "Medium"),
    _loc("IN-ROI-10", "Raipur",        "Raipur",      "Chhattisgarh",    "Central Plateau",     21.2514, 81.6296, 298, "Red",           ""),
    _loc("IN-ROI-11", "Bhubaneswar",   "Khordha",     "Odisha",          "Eastern Coastal",     20.2961, 85.8245, 45,  "Laterite",      ""),
    _loc("IN-ROI-12", "Chandigarh",    "Chandigarh",  "Chandigarh",      "Indo-Gangetic Plain", 30.7333, 76.7794, 321, "Alluvial",      ""),
    _loc("IN-ROI-13", "Coimbatore",    "Coimbatore",  "Tamil Nadu",      "Western Ghats",       11.0168, 76.9558, 411, "Red",           ""),
    _loc("IN-ROI-14", "Vijayawada",    "Krishna",     "Andhra Pradesh",  "Coastal Plain",       16.5062, 80.6480, 12,  "Alluvial",      ""),
    _loc("IN-ROI-15", "Jodhpur",       "Jodhpur",     "Rajasthan",       "Arid Western",        26.2389, 73.0243, 231, "Sandy",         ""),
]


# ─── MH-only seed set (hand-curated 85 locations) ───
MH_SEEDS = VIDARBHA + MARATHWADA + WESTERN_MH + KONKAN + NORTH_MH


def _zone_prefix(zone: str) -> str:
    """Zone → ID prefix used for synthetic locations."""
    return {
        "Vidarbha":            "MH-VID",
        "Marathwada":          "MH-MTH",
        "Western Maharashtra": "MH-WST",
        "Konkan":              "MH-KNK",
        "North Maharashtra":   "MH-NTH",
    }.get(zone, "MH-UNK")


def _perturb_seed(seed: dict, synthetic_idx: int, rng: random.Random) -> dict:
    """Produce one synthetic location by jittering a seed (lat/lon/alt/irrigation).

    Soil type and agro_zone are preserved (they are inherent to the district).
    Drainage is re-sampled from the SOIL_PROPERTIES entry so that synthetic
    points vary in sub-type (e.g. Deep vs Medium Black) around the parent.
    """
    zone = seed["agro_zone"]
    soil_type = seed["soil_type"]
    # Re-sample subtype for Black (Regur) so Deep/Medium/Shallow are diverse
    if soil_type == "Black (Regur)":
        subtype = rng.choice(["Deep", "Medium", "Shallow"])
    else:
        subtype = seed.get("soil_subtype", "")

    # Irrigation: bias per-zone, but respect if seed is strongly irrigated
    irr_prob = IRRIGATION_PROB_BY_ZONE.get(zone, 0.25)
    # Seeds already marked irrigated stay irrigated with 80% probability
    if seed.get("irrigation_available", 0) == 1:
        irrigated = int(rng.random() < 0.80)
    else:
        irrigated = int(rng.random() < irr_prob)
    if irrigated:
        ws_pool = WATER_SOURCES_BY_IRR.get(zone, ["borewell"])
        water_source = rng.choice(ws_pool)
    else:
        water_source = "rainfed"

    lat = round(seed["lat"] + rng.uniform(-JITTER_LAT, JITTER_LAT), 4)
    lon = round(seed["lon"] + rng.uniform(-JITTER_LON, JITTER_LON), 4)
    alt = max(3, int(seed["altitude_m"] + rng.randint(-JITTER_ALT, JITTER_ALT)))
    loc_id = f"{_zone_prefix(zone)}-S{synthetic_idx:03d}"
    city = f"{seed['city']}-Satellite-{synthetic_idx:03d}"
    return _loc(
        loc_id, city, seed["district"], seed["state"], zone,
        lat, lon, alt, soil_type, subtype,
        irrigation=irrigated, water_source=water_source,
    )


def _expand_zone(seeds: list[dict], target: int, rng: random.Random,
                 idx_start: int) -> tuple[list[dict], int]:
    """Expand a zone from its seed list up to `target` locations."""
    expanded = list(seeds)
    needed = max(0, target - len(seeds))
    idx = idx_start
    # Distribute synthetic points evenly across seeds
    if needed > 0 and seeds:
        per_seed = needed // len(seeds)
        remainder = needed - per_seed * len(seeds)
        for s_i, seed in enumerate(seeds):
            n_local = per_seed + (1 if s_i < remainder else 0)
            for _ in range(n_local):
                expanded.append(_perturb_seed(seed, idx, rng))
                idx += 1
    return expanded, idx


def _build_mh_locations() -> list[dict]:
    """Build the full ~300-location MH set by expanding seed zones."""
    rng = random.Random(EXPANSION_SEED)
    all_loc: list[dict] = []
    idx = 1
    zone_seeds = {
        "Vidarbha":            VIDARBHA,
        "Marathwada":          MARATHWADA,
        "Western Maharashtra": WESTERN_MH,
        "Konkan":               KONKAN,
        "North Maharashtra":    NORTH_MH,
    }
    for zone, target in ZONE_TARGETS.items():
        zone_locs, idx = _expand_zone(zone_seeds[zone], target, rng, idx)
        all_loc.extend(zone_locs)
    return all_loc


# ─── Default export: Maharashtra-only, ~300 locations ───
ALL_LOCATIONS = _build_mh_locations()


def generate_locations_json() -> list[dict]:
    """Generate locations_100.json (MH-only expanded set) and return the list."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    locations = ALL_LOCATIONS
    print(f"Total locations: {len(locations)}")

    # Distribution summary
    regions = {}
    soil_types = {}
    irr_count = 0
    for loc in locations:
        zone = loc["agro_zone"]
        soil = loc["soil_type"]
        regions[zone] = regions.get(zone, 0) + 1
        soil_types[soil] = soil_types.get(soil, 0) + 1
        irr_count += loc.get("irrigation_available", 0)

    print("\n── Region Distribution ──")
    for zone, count in sorted(regions.items(), key=lambda x: -x[1]):
        print(f"  {zone}: {count}")

    print("\n── Soil Type Distribution ──")
    for soil, count in sorted(soil_types.items(), key=lambda x: -x[1]):
        pct = count / len(locations) * 100
        print(f"  {soil}: {count} ({pct:.0f}%)")

    # Maharashtra count
    mh_count = sum(1 for loc in locations if loc["state"] == "Maharashtra")
    print(f"\nMaharashtra locations: {mh_count}/{len(locations)} ({mh_count/len(locations)*100:.0f}%)")
    print(f"Irrigated locations:  {irr_count}/{len(locations)} ({irr_count/len(locations)*100:.0f}%)")

    # Write JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {OUTPUT_FILE}")
    return locations


def get_locations_by_region(region: str) -> list[dict]:
    """Get locations for a specific region/agro_zone."""
    return [loc for loc in ALL_LOCATIONS if loc["agro_zone"] == region]


if __name__ == "__main__":
    generate_locations_json()
