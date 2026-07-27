"""
Crop Profiles — Agronomic requirements for 33+ crops across Kharif, Rabi, Zaid, Annual seasons.
Used by sensor_data_generator.py and crop_label_validator.py.

Each crop defines: soil affinities, pH range, NPK requirements, rainfall/temp needs.
NPK bands have been tightened (v2026_05) for ML separability while remaining
agronomically valid; overlapping Kharif/Rabi clusters now differ by ≥15 mg/kg
on their dominant nutrient to reduce Bayes error.

Source: PLAN.md Section 5 + ICAR Maharashtra guidelines.
"""

# Crop family taxonomy
CROP_FAMILIES = {
    "Cereal":    ["Rice", "Wheat", "Jowar (Kharif)", "Rabi Jowar", "Bajra", "Maize", "Ragi"],
    "Legume":    ["Soybean", "Chickpea (Gram)", "Pigeonpea (Tur)", "Green Gram", "Black Gram", "Lentil"],
    "Oilseed":   ["Groundnut", "Sunflower", "Linseed", "Safflower", "Sesame", "Mustard"],
    "Cash":      ["Cotton", "Sugarcane", "Grape", "Onion", "Turmeric"],
    "Vegetable": ["Tomato", "Chilli", "Okra", "Brinjal", "Potato", "Coriander"],
    "Plantation":["Banana", "Mango", "Pomegranate", "Cashew", "Coconut"],
}

# Reverse lookup: crop_name -> crop_family
CROP_TO_FAMILY = {}
for family, crops in CROP_FAMILIES.items():
    for crop in crops:
        CROP_TO_FAMILY[crop] = family


# ─── KHARIF CROPS (June – October) ───
# NPK bands tightened (v2026_05): each crop separated by ≥15 mg/kg on its
# dominant nutrient from its nearest Kharif neighbour (see PLAN.md §5.1).
KHARIF_CROPS = {
    "Soybean": {
        # Legume — modest N (fixes its own), moderate P, low K
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (6.0, 7.2), "n_range": (70, 110), "p_range": (45, 70), "k_range": (45, 80),
        "rainfall_mm": (650, 1000), "temp_range": (22, 30),
    },
    "Cotton": {
        # Cash crop — high N, high K, alkaline-tolerant
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (7.0, 8.0), "n_range": (140, 200), "p_range": (55, 85), "k_range": (150, 210),
        "rainfall_mm": (500, 800), "temp_range": (24, 34),
    },
    "Jowar (Kharif)": {
        # Coarse cereal — mid N, low P, mid K, dryland
        "soil_affinity": ["Black (Regur)", "Red"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (6.5, 7.8), "n_range": (90, 130), "p_range": (25, 45), "k_range": (85, 120),
        "rainfall_mm": (400, 650), "temp_range": (26, 34),
    },
    "Bajra": {
        # Millet — low N, very low P, low K, sandy/arid
        "soil_affinity": ["Sandy", "Red"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (6.0, 7.5), "n_range": (60, 95), "p_range": (18, 38), "k_range": (50, 80),
        "rainfall_mm": (280, 550), "temp_range": (27, 36),
    },
    "Ragi": {
        # Finger millet — low-input, acidic-tolerant, hilly/Konkan
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Sandy"],
        "ph_range": (5.0, 6.5), "n_range": (40, 75), "p_range": (20, 40), "k_range": (35, 65),
        "rainfall_mm": (450, 800), "temp_range": (20, 30),
    },
    "Maize": {
        # Cereal — highest N, high P, very high K
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (5.8, 7.2), "n_range": (165, 220), "p_range": (70, 100), "k_range": (155, 200),
        "rainfall_mm": (520, 900), "temp_range": (21, 32),
    },
    "Rice": {
        # Paddy — high N, mid P, high K, flood-irrigated, acidic
        "soil_affinity": ["Alluvial", "Clay"],
        "soil_secondary": ["Laterite", "Black (Regur)"],
        "ph_range": (5.2, 6.8), "n_range": (160, 215), "p_range": (55, 78), "k_range": (100, 145),
        "rainfall_mm": (1100, 2200), "temp_range": (23, 35),
    },
    "Groundnut": {
        # Oilseed — low N (legume), mid P, mid K, sandy
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (5.8, 6.8), "n_range": (35, 70), "p_range": (45, 70), "k_range": (85, 130),
        "rainfall_mm": (500, 800), "temp_range": (22, 33),
    },
    "Pigeonpea (Tur)": {
        # Legume — very low N, mid P, very high K, alkaline
        "soil_affinity": ["Black (Regur)", "Red"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (7.0, 7.8), "n_range": (30, 55), "p_range": (55, 80), "k_range": (120, 160),
        "rainfall_mm": (500, 850), "temp_range": (25, 35),
    },
    "Green Gram": {
        # Legume — very low N, very low P, very low K, hot/dry
        "soil_affinity": ["Sandy", "Laterite"],
        "soil_secondary": ["Red"],
        "ph_range": (5.5, 6.5), "n_range": (18, 40), "p_range": (22, 40), "k_range": (35, 60),
        "rainfall_mm": (350, 600), "temp_range": (28, 38),
    },
    "Sesame": {
        # Oilseed — mid-low N, very low P, mid-low K, hot/dry
        "soil_affinity": ["Sandy", "Red"],
        "soil_secondary": ["Alluvial", "Laterite"],
        "ph_range": (5.8, 7.0), "n_range": (40, 75), "p_range": (28, 48), "k_range": (40, 75),
        "rainfall_mm": (300, 550), "temp_range": (26, 38),
    },
    "Turmeric": {
        # Rhizome cash crop — high N, high P, very high K, acidic, wet
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Laterite"],
        "ph_range": (5.0, 6.5), "n_range": (110, 165), "p_range": (65, 105), "k_range": (125, 185),
        "rainfall_mm": (850, 1600), "temp_range": (20, 32),
    },
    "Black Gram": {
        # Legume — extremely low N, extremely low P, extremely low K
        "soil_affinity": ["Alluvial", "Laterite"],
        "soil_secondary": ["Red", "Sandy"],
        "ph_range": (6.5, 7.5), "n_range": (12, 35), "p_range": (18, 35), "k_range": (28, 55),
        "rainfall_mm": (450, 750), "temp_range": (27, 36),
    },
}


# ─── RABI CROPS (November – February) ───
# Tightened (v2026_05): each crop separated by ≥15 mg/kg on its dominant nutrient.
RABI_CROPS = {
    "Wheat": {
        # Cereal — highest Rabi N, high P, mid-high K
        "soil_affinity": ["Alluvial"],
        "soil_secondary": ["Black (Regur)", "Clay"],
        "ph_range": (6.2, 7.5), "n_range": (130, 185), "p_range": (85, 125), "k_range": (125, 175),
        "rainfall_mm": (250, 550), "temp_range": (10, 24),
    },
    "Chickpea (Gram)": {
        # Legume — very low N, low P, mid K, alkaline-tolerant
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red", "Alluvial"],
        "ph_range": (6.8, 8.0), "n_range": (18, 38), "p_range": (42, 62), "k_range": (90, 120),
        "rainfall_mm": (200, 480), "temp_range": (12, 25),
    },
    "Rabi Jowar": {
        # Cereal — mid N, low-mid P, mid K, heavy black soil
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (7.0, 8.0), "n_range": (80, 115), "p_range": (30, 50), "k_range": (75, 110),
        "rainfall_mm": (200, 450), "temp_range": (14, 28),
    },
    "Sunflower": {
        # Oilseed — mid N, mid-high P, high K
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (75, 115), "p_range": (62, 90), "k_range": (100, 145),
        "rainfall_mm": (250, 500), "temp_range": (16, 30),
    },
    "Linseed": {
        # Oilseed — mid-low N, low P, low-mid K, acidic-tolerant
        "soil_affinity": ["Red", "Alluvial"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (5.8, 7.0), "n_range": (55, 90), "p_range": (28, 48), "k_range": (55, 90),
        "rainfall_mm": (200, 430), "temp_range": (10, 24),
    },
    "Safflower": {
        # Oilseed — low-mid N, mid P, mid K, alkaline
        "soil_affinity": ["Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (7.0, 8.0), "n_range": (55, 85), "p_range": (38, 58), "k_range": (80, 120),
        "rainfall_mm": (200, 430), "temp_range": (16, 30),
    },
    "Mustard": {
        # Oilseed — mid N, mid P, mid-high K, alkaline-tolerant
        "soil_affinity": ["Alluvial", "Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (6.5, 7.8), "n_range": (95, 130), "p_range": (45, 70), "k_range": (95, 135),
        "rainfall_mm": (250, 450), "temp_range": (10, 25),
    },
    "Onion": {
        # Bulb vegetable — mid-high N, high P, high K, neutral pH
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Alluvial", "Black (Regur)"],
        "ph_range": (6.0, 7.2), "n_range": (105, 145), "p_range": (80, 115), "k_range": (130, 175),
        "rainfall_mm": (300, 600), "temp_range": (14, 25),
    },
    "Potato": {
        # Tuber vegetable — highest Rabi N, highest Rabi P, very high K, slightly acidic
        "soil_affinity": ["Alluvial", "Red"],
        "soil_secondary": ["Sandy"],
        "ph_range": (5.5, 6.5), "n_range": (170, 225), "p_range": (140, 185), "k_range": (185, 240),
        "rainfall_mm": (300, 550), "temp_range": (12, 22),
    },
    "Coriander": {
        # Spice/vegetable — low N, low P, low-mid K, short-duration
        "soil_affinity": ["Alluvial", "Red"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.2, 7.3), "n_range": (35, 65), "p_range": (20, 40), "k_range": (55, 85),
        "rainfall_mm": (200, 400), "temp_range": (14, 26),
    },
    "Lentil": {
        # Legume — very low N, low P, very low K
        "soil_affinity": ["Alluvial", "Red"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.3), "n_range": (15, 35), "p_range": (28, 48), "k_range": (40, 70),
        "rainfall_mm": (200, 430), "temp_range": (10, 24),
    },
}


# ─── ZAID CROPS (March – May) ───
# NOTE: Profiles deliberately spaced for ML separability while remaining agronomically valid.
ZAID_CROPS = {
    "Okra": {
        "soil_affinity": ["Sandy", "Laterite"],
        "soil_secondary": ["Red"],
        "ph_range": (5.5, 6.5), "n_range": (35, 65), "p_range": (15, 35), "k_range": (25, 55),
        "rainfall_mm": (50, 200), "temp_range": (32, 42),
    },
    "Tomato": {
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (5.0, 6.2), "n_range": (150, 220), "p_range": (90, 130), "k_range": (160, 240),
        "rainfall_mm": (200, 450), "temp_range": (18, 28),
    },
    "Chilli": {
        "soil_affinity": ["Red", "Sandy"],
        "soil_secondary": ["Laterite", "Alluvial"],
        "ph_range": (6.0, 7.5), "n_range": (50, 90), "p_range": (25, 50), "k_range": (40, 80),
        "rainfall_mm": (50, 180), "temp_range": (30, 40),
    },
    "Brinjal": {
        "soil_affinity": ["Alluvial", "Clay"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.2, 7.2), "n_range": (110, 160), "p_range": (65, 95), "k_range": (110, 170),
        "rainfall_mm": (250, 500), "temp_range": (22, 32),
    },
}


# ─── ANNUAL / PERENNIAL CROPS (multi-season, year-round) ───
# These crops do not fit a single short-window season because they are either
# perennials (fruit trees, vines, plantations) or 12–18-month crops (Sugarcane,
# Banana). They are given their own season tag "Annual" and bypass the hard
# Kharif/Rabi/Zaid filter during inference — the model is allowed to recommend
# them regardless of the query month.
ANNUAL_CROPS = {
    "Sugarcane": {
        # 12–18 month crop — very high N, high P, very high K, high water
        "soil_affinity": ["Alluvial", "Black (Regur)"],
        "soil_secondary": ["Red"],
        "ph_range": (6.5, 7.8), "n_range": (200, 300), "p_range": (90, 150), "k_range": (200, 300),
        "rainfall_mm": (1500, 2500), "temp_range": (22, 35),
    },
    "Banana": {
        # 10–15 month perennial — very high N, mid P, very high K, tropical
        "soil_affinity": ["Alluvial", "Red"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.0, 7.5), "n_range": (220, 320), "p_range": (60, 100), "k_range": (300, 450),
        "rainfall_mm": (1200, 2200), "temp_range": (22, 35),
    },
    "Grape": {
        # Perennial vine — high N, high P, very high K, Nashik/Sangli
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Black (Regur)"],
        "ph_range": (6.3, 7.5), "n_range": (100, 140), "p_range": (80, 120), "k_range": (195, 260),
        "rainfall_mm": (250, 600), "temp_range": (15, 35),
    },
    "Mango": {
        # Perennial tree — mid N, mid P, mid-high K, Konkan/Ratnagiri
        "soil_affinity": ["Red", "Laterite"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (5.5, 7.5), "n_range": (70, 110), "p_range": (45, 75), "k_range": (110, 160),
        "rainfall_mm": (750, 2500), "temp_range": (20, 38),
    },
    "Pomegranate": {
        # Perennial shrub — mid N, mid P, mid K, arid (Solapur/Nashik)
        "soil_affinity": ["Red", "Black (Regur)"],
        "soil_secondary": ["Laterite"],
        "ph_range": (6.5, 7.8), "n_range": (55, 90), "p_range": (35, 60), "k_range": (90, 135),
        "rainfall_mm": (400, 800), "temp_range": (18, 38),
    },
    "Cashew": {
        # Perennial tree — low N, low P, low K, acidic, Konkan coast
        "soil_affinity": ["Laterite", "Red"],
        "soil_secondary": ["Sandy"],
        "ph_range": (4.8, 6.2), "n_range": (40, 70), "p_range": (20, 40), "k_range": (55, 95),
        "rainfall_mm": (1200, 3000), "temp_range": (22, 36),
    },
    "Coconut": {
        # Perennial palm — mid N, low P, very high K, coastal Konkan
        "soil_affinity": ["Sandy", "Laterite"],
        "soil_secondary": ["Alluvial"],
        "ph_range": (5.5, 7.5), "n_range": (80, 130), "p_range": (25, 50), "k_range": (240, 360),
        "rainfall_mm": (1500, 3500), "temp_range": (24, 36),
    },
}


# ─── Combined lookup ───
ALL_CROPS = {
    "Kharif": KHARIF_CROPS,
    "Rabi": RABI_CROPS,
    "Zaid": ZAID_CROPS,
    "Annual": ANNUAL_CROPS,
}

# Reverse lookup: crop name → season (for hard season filtering in inference)
CROP_TO_SEASON = {}
for _season, _crops in ALL_CROPS.items():
    for _crop_name in _crops:
        CROP_TO_SEASON[_crop_name] = _season

# Season month mappings. "Annual" covers all 12 months (perennials / long crops).
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9, 10],
    "Rabi": [11, 12, 1, 2],
    "Zaid": [3, 4, 5],
    "Annual": list(range(1, 13)),
}

# Crops that must bypass the hard season filter in inference (perennials + long-duration).
ANNUAL_CROP_NAMES = set(ANNUAL_CROPS.keys())

# Soil profiles from AgroSensor (for NPK baselines and sensor generation)
SOIL_PROFILES = {
    "Alluvial":      {"water_retention": 0.65, "drainage_rate": 0.6,  "ph_base": 7.0, "ph_range": (6.5, 8.0), "ec_base": 1100, "ec_range": (400, 2200),  "n_base": 140, "p_base": 90, "k_base": 170},
    "Black (Regur)": {"water_retention": 0.85, "drainage_rate": 0.3,  "ph_base": 7.8, "ph_range": (7.0, 8.5), "ec_base": 1400, "ec_range": (600, 2800),  "n_base": 110, "p_base": 70, "k_base": 200},
    "Red":           {"water_retention": 0.40, "drainage_rate": 0.75, "ph_base": 5.8, "ph_range": (4.5, 6.5), "ec_base": 800,  "ec_range": (200, 1500),  "n_base": 80,  "p_base": 50, "k_base": 100},
    "Laterite":      {"water_retention": 0.35, "drainage_rate": 0.80, "ph_base": 5.5, "ph_range": (4.5, 6.0), "ec_base": 600,  "ec_range": (150, 1200),  "n_base": 60,  "p_base": 40, "k_base": 80},
    "Sandy":         {"water_retention": 0.20, "drainage_rate": 0.90, "ph_base": 6.5, "ph_range": (5.5, 7.5), "ec_base": 500,  "ec_range": (100, 1000),  "n_base": 50,  "p_base": 30, "k_base": 60},
    "Clay":          {"water_retention": 0.90, "drainage_rate": 0.20, "ph_base": 7.2, "ph_range": (6.5, 8.5), "ec_base": 1500, "ec_range": (700, 3000),  "n_base": 130, "p_base": 80, "k_base": 180},
}


# ─── Regional Crop Dominance (agro_zone → season → crops with boost weights) ───
# Based on Maharashtra State Agriculture Census cropping area data.
# These biases ensure the generator produces region-realistic crop distributions.
REGIONAL_CROP_DOMINANCE = {
    "Vidarbha": {
        "Kharif": {"Cotton": 2.5, "Soybean": 2.5, "Pigeonpea (Tur)": 2.0, "Jowar (Kharif)": 1.3, "Green Gram": 1.2},
        "Rabi":   {"Chickpea (Gram)": 2.5, "Wheat": 1.8, "Rabi Jowar": 1.5, "Safflower": 1.5, "Linseed": 1.3, "Mustard": 1.2},
        "Zaid":   {"Okra": 1.2, "Chilli": 1.0},
        "Annual": {"Sugarcane": 1.2, "Mango": 1.1},
    },
    "Marathwada": {
        "Kharif": {"Cotton": 2.0, "Soybean": 2.0, "Bajra": 1.5, "Pigeonpea (Tur)": 1.8, "Jowar (Kharif)": 1.5},
        "Rabi":   {"Rabi Jowar": 2.5, "Chickpea (Gram)": 2.0, "Sunflower": 1.5, "Safflower": 1.5, "Coriander": 1.2},
        "Zaid":   {"Okra": 1.0, "Chilli": 1.0},
        "Annual": {"Pomegranate": 2.0, "Sugarcane": 1.3, "Mango": 1.0},
    },
    "Western Maharashtra": {
        "Kharif": {"Rice": 1.8, "Soybean": 1.5, "Groundnut": 1.3, "Maize": 1.3, "Turmeric": 1.2},
        "Rabi":   {"Wheat": 2.0, "Onion": 2.5, "Chickpea (Gram)": 1.5, "Potato": 1.5},
        "Zaid":   {"Tomato": 1.5, "Okra": 1.2, "Brinjal": 1.2},
        "Annual": {"Sugarcane": 2.5, "Grape": 2.0, "Banana": 1.8, "Pomegranate": 1.5},
    },
    "Konkan": {
        "Kharif": {"Rice": 3.0, "Groundnut": 1.5, "Sesame": 1.3, "Black Gram": 1.3, "Ragi": 1.8, "Turmeric": 1.3},
        "Rabi":   {"Lentil": 1.5, "Chickpea (Gram)": 1.0, "Coriander": 1.1},
        "Zaid":   {"Okra": 1.3, "Brinjal": 1.2},
        "Annual": {"Mango": 3.0, "Cashew": 3.0, "Coconut": 2.5, "Banana": 1.5},
    },
    "North Maharashtra": {
        "Kharif": {"Bajra": 2.0, "Maize": 1.8, "Cotton": 1.5, "Groundnut": 1.5},
        "Rabi":   {"Onion": 2.5, "Wheat": 2.0, "Chickpea (Gram)": 1.5, "Mustard": 1.3, "Potato": 1.2},
        "Zaid":   {"Tomato": 1.5, "Chilli": 1.2, "Okra": 1.0},
        "Annual": {"Grape": 2.5, "Sugarcane": 1.5, "Banana": 1.8, "Pomegranate": 1.3},
    },
}


# ─── EC-sensitive crops (penalised when EC > threshold) ───
# Crops that do poorly in saline/alkaline soils with high electrical conductivity.
EC_SENSITIVE_CROPS = {
    # crop_name: max_ec_threshold (μS/cm) — above this, crop is penalised
    "Soybean": 1600,
    "Green Gram": 1200,
    "Black Gram": 1400,
    "Sesame": 1200,
    "Lentil": 1200,
    "Okra": 1400,
    "Chilli": 1400,
    "Tomato": 1600,
    "Brinjal": 1800,
    "Groundnut": 1500,
    "Sunflower": 2000,
    "Maize": 1800,
    "Ragi": 1400,
    "Potato": 1500,
    "Coriander": 1500,
    "Cashew": 1400,
    "Banana": 1800,
    "Mango": 1800,
}

# EC-tolerant crops (can handle higher EC)
EC_TOLERANT_CROPS = {
    "Cotton": 3500,
    "Sugarcane": 3000,
    "Wheat": 2800,
    "Bajra": 3000,
    "Rabi Jowar": 2800,
    "Jowar (Kharif)": 2800,
    "Safflower": 2500,
    "Rice": 2500,
    "Mustard": 2800,
    "Pomegranate": 3200,
    "Coconut": 2800,
}


# ─── Soil-Crop Incompatibility (hard penalties for the guardrail layer) ───
# Crops that should NEVER be recommended for certain soil types.
SOIL_CROP_INCOMPATIBLE = {
    "Black (Regur)": ["Chilli", "Okra", "Groundnut", "Sesame", "Cashew", "Coconut"],  # heavy clay — bad for light-soil crops
    "Clay":          ["Chilli", "Okra", "Groundnut", "Sesame", "Cashew", "Coconut", "Potato"],
    "Sandy":         ["Rice", "Sugarcane", "Banana"],                                  # can't retain water
    "Laterite":      ["Rice", "Sugarcane", "Cotton", "Wheat", "Potato"],               # poor fertility
}


# ─── Drainage-Crop Compatibility ───
# Crops that THRIVE under poor drainage (waterlogged / standing water)
DRAINAGE_TOLERANT_CROPS = {"Rice", "Sugarcane"}

# Crops that FAIL under poor drainage (need well-drained soil)
DRAINAGE_SENSITIVE_CROPS = {
    "Chilli", "Brinjal", "Tomato", "Okra",        # vegetables — root rot
    "Onion", "Potato",                             # bulb/tuber rot in waterlogging
    "Groundnut", "Sesame",                         # oilseeds — need drainage
    "Soybean",                                     # moderate tolerance but not poor
    "Cotton",                                      # boll rot in standing water
    "Sunflower", "Safflower", "Mustard",           # oilseeds
    "Jowar (Kharif)", "Rabi Jowar", "Bajra", "Ragi",  # dryland cereals — waterlogging intolerant
    "Maize",                                       # root suffocation in standing water
    "Grape", "Pomegranate", "Mango", "Cashew",     # perennials — root rot in waterlogged soils
    "Coriander",                                   # shallow roots — rot easily
}

# Valid drainage values per soil type (for input consistency validation)
# Based on SOIL_PROPERTIES from location_generator.py
SOIL_DRAINAGE_VALID = {
    "Sandy":         ["Good", "Excessive"],           # Sandy cannot have Poor drainage
    "Laterite":      ["Good", "Excessive"],
    "Red":           ["Good", "Moderate"],
    "Alluvial":      ["Good", "Moderate"],
    "Black (Regur)": ["Poor", "Moderate"],            # heavy clay → poor or moderate
    "Clay":          ["Very Poor", "Poor", "Moderate"],
}



# ─── Agro Zone Geographic Bounding Boxes ───
# Approximate lat/lon ranges for each Maharashtra agro zone.
# Used for input validation: flagging zone-location mismatches.
AGRO_ZONE_BOUNDS = {
    "Vidarbha":              {"lat": (19.5, 22.0), "lon": (76.0, 80.5)},
    "Marathwada":            {"lat": (17.5, 20.5), "lon": (74.5, 77.5)},
    "Western Maharashtra":   {"lat": (15.5, 19.5), "lon": (73.5, 76.0)},
    "Konkan":                {"lat": (15.5, 20.0), "lon": (72.5, 74.0)},
    "North Maharashtra":     {"lat": (19.5, 22.0), "lon": (72.5, 76.0)},
    "Northern Maharashtra":  {"lat": (19.5, 22.0), "lon": (72.5, 76.0)},
}

# ─── Crop Decision Engine Constants (T8) ───

CROP_YIELD_BENCHMARKS = {
    # Crop: max_t_ha, avg_t_ha, unit, duration_days
    "Cotton": {"max_t_ha": 3.0, "avg_t_ha": 1.2, "unit": "t lint/ha", "duration_days": 150},
    "Soybean": {"max_t_ha": 3.5, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 100},
    "Wheat": {"max_t_ha": 5.5, "avg_t_ha": 3.2, "unit": "t/ha", "duration_days": 120},
    "Rice": {"max_t_ha": 7.5, "avg_t_ha": 3.5, "unit": "t/ha", "duration_days": 130},
    "Jowar (Kharif)": {"max_t_ha": 4.0, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 110},
    "Rabi Jowar": {"max_t_ha": 3.5, "avg_t_ha": 1.2, "unit": "t/ha", "duration_days": 110},
    "Bajra": {"max_t_ha": 3.0, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 90},
    "Maize": {"max_t_ha": 8.0, "avg_t_ha": 3.5, "unit": "t/ha", "duration_days": 110},
    "Ragi": {"max_t_ha": 3.5, "avg_t_ha": 1.5, "unit": "t/ha", "duration_days": 120},
    "Chickpea (Gram)": {"max_t_ha": 2.5, "avg_t_ha": 1.1, "unit": "t/ha", "duration_days": 110},
    "Pigeonpea (Tur)": {"max_t_ha": 2.5, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 160},
    "Green Gram": {"max_t_ha": 1.5, "avg_t_ha": 0.6, "unit": "t/ha", "duration_days": 65},
    "Black Gram": {"max_t_ha": 1.5, "avg_t_ha": 0.6, "unit": "t/ha", "duration_days": 70},
    "Lentil": {"max_t_ha": 2.0, "avg_t_ha": 0.9, "unit": "t/ha", "duration_days": 110},
    "Groundnut": {"max_t_ha": 3.5, "avg_t_ha": 1.6, "unit": "t/ha", "duration_days": 110},
    "Sunflower": {"max_t_ha": 2.5, "avg_t_ha": 0.9, "unit": "t/ha", "duration_days": 100},
    "Linseed": {"max_t_ha": 1.5, "avg_t_ha": 0.5, "unit": "t/ha", "duration_days": 120},
    "Safflower": {"max_t_ha": 2.0, "avg_t_ha": 0.8, "unit": "t/ha", "duration_days": 130},
    "Sesame": {"max_t_ha": 1.2, "avg_t_ha": 0.4, "unit": "t/ha", "duration_days": 90},
    "Mustard": {"max_t_ha": 2.5, "avg_t_ha": 1.2, "unit": "t/ha", "duration_days": 110},
    "Sugarcane": {"max_t_ha": 120.0, "avg_t_ha": 80.0, "unit": "t/ha", "duration_days": 360},
    "Grape": {"max_t_ha": 35.0, "avg_t_ha": 22.0, "unit": "t/ha", "duration_days": 180},
    "Onion": {"max_t_ha": 30.0, "avg_t_ha": 16.0, "unit": "t/ha", "duration_days": 120},
    "Turmeric": {"max_t_ha": 25.0, "avg_t_ha": 12.0, "unit": "t/ha", "duration_days": 240},
    "Tomato": {"max_t_ha": 60.0, "avg_t_ha": 25.0, "unit": "t/ha", "duration_days": 130},
    "Chilli": {"max_t_ha": 15.0, "avg_t_ha": 6.0, "unit": "t/ha (dry)", "duration_days": 150},
    "Okra": {"max_t_ha": 18.0, "avg_t_ha": 10.0, "unit": "t/ha", "duration_days": 100},
    "Brinjal": {"max_t_ha": 45.0, "avg_t_ha": 20.0, "unit": "t/ha", "duration_days": 140},
    "Potato": {"max_t_ha": 40.0, "avg_t_ha": 22.0, "unit": "t/ha", "duration_days": 110},
    "Coriander": {"max_t_ha": 1.5, "avg_t_ha": 0.7, "unit": "t/ha (seed)", "duration_days": 100},
    "Banana": {"max_t_ha": 70.0, "avg_t_ha": 40.0, "unit": "t/ha", "duration_days": 300},
    "Mango": {"max_t_ha": 12.0, "avg_t_ha": 6.0, "unit": "t/ha", "duration_days": 365},
    "Pomegranate": {"max_t_ha": 20.0, "avg_t_ha": 12.0, "unit": "t/ha", "duration_days": 300},
    "Cashew": {"max_t_ha": 2.5, "avg_t_ha": 1.0, "unit": "t/ha", "duration_days": 365},
    "Coconut": {"max_t_ha": 15000, "avg_t_ha": 8000, "unit": "nuts/ha", "duration_days": 365},
}

CROP_MSP = {
    # Approx MSP or market price per tonne in INR (2025/2026 estimates)
    "Cotton": 75000,
    "Soybean": 48000,
    "Wheat": 22750,
    "Rice": 23000,
    "Jowar (Kharif)": 31800,
    "Rabi Jowar": 32250,
    "Bajra": 25000,
    "Maize": 20900,
    "Ragi": 38460,
    "Chickpea (Gram)": 54400,
    "Pigeonpea (Tur)": 70000,
    "Green Gram": 85580,
    "Black Gram": 69500,
    "Lentil": 64250,
    "Groundnut": 63770,
    "Sunflower": 67600,
    "Linseed": 58000,
    "Safflower": 58000,
    "Sesame": 86350,
    "Mustard": 56500,
    "Sugarcane": 3400, # Per tonne
    "Grape": 40000, # Market price
    "Onion": 15000, # Market price
    "Turmeric": 140000,
    "Tomato": 12000,
    "Chilli": 180000, # Dry
    "Okra": 20000,
    "Brinjal": 15000,
    "Potato": 12000,
    "Coriander": 85000,
    "Banana": 15000,
    "Mango": 60000,
    "Pomegranate": 80000,
    "Cashew": 100000,
    "Coconut": 15, # Per nut
}

CROP_AGRONOMY = {
    "Cotton": {
        "sow_months": [6, 7],
        "harvest_months": [11, 12, 1],
        "seed_rate": "2.5 kg/ha (Bt hybrids)",
        "spacing": "90×30 cm (120×45 for drip)",
        "irrigation_count": "6–8",
        "irrigation_mm": 50,
        "pest_watch": ["Bollworm", "Jassids", "Whiteflies", "Aphids"],
        "fert_splits": ["25% basal", "50% at square formation", "25% at boll formation"],
        "next_crop": "Chickpea or Wheat",
    },
    "Soybean": {
        "sow_months": [6, 7],
        "harvest_months": [9, 10],
        "seed_rate": "65-70 kg/ha",
        "spacing": "45×15 cm",
        "irrigation_count": "Rainfed (1-2 protective if dry)",
        "irrigation_mm": 50,
        "pest_watch": ["Girdle beetle", "Stem fly", "Defoliators"],
        "fert_splits": ["100% basal (N-P-K) + Sulphur"],
        "next_crop": "Wheat or Chickpea",
    },
    "Wheat": {
        "sow_months": [11, 12],
        "harvest_months": [3, 4],
        "seed_rate": "100-125 kg/ha",
        "spacing": "22.5 cm (row)",
        "irrigation_count": "4–6",
        "irrigation_mm": 60,
        "pest_watch": ["Rusts (Brown/Yellow)", "Aphids", "Termites"],
        "fert_splits": ["50% N + 100% P&K basal", "50% N at Crown Root Initiation (CRI)"],
        "next_crop": "Green gram or Fallow",
    },
    "Rice": {
        "sow_months": [6, 7],
        "harvest_months": [10, 11],
        "seed_rate": "40-50 kg/ha (transplanted)",
        "spacing": "20×15 cm",
        "irrigation_count": "Continuous flooding / AWD",
        "irrigation_mm": "Maintain 5cm standing water",
        "pest_watch": ["Stem borer", "Leaf folder", "Blast", "BPH"],
        "fert_splits": ["50% N, 100% P&K basal", "25% N tillering", "25% N panicle initiation"],
        "next_crop": "Gram or Lentil",
    },
    "Potato": {
        "sow_months": [10, 11],
        "harvest_months": [2, 3],
        "seed_rate": "2.5-3.0 t/ha (tubers)",
        "spacing": "60×20 cm",
        "irrigation_count": "7–10 (light & frequent)",
        "irrigation_mm": 40,
        "pest_watch": ["Late Blight", "Aphids", "White grubs"],
        "fert_splits": ["50% N, 100% P&K basal", "50% N at earthing up (30 DAS)"],
        "next_crop": "Zaid vegetables or Green gram",
    },
    "Coconut": {
        "sow_months": [6, 7, 8],
        "harvest_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "seed_rate": "140-150 seedlings/ha",
        "spacing": "7.5×7.5 m",
        "irrigation_count": "Summer months (Drip preferred)",
        "irrigation_mm": "50-60 liters/palm/day in summer",
        "pest_watch": ["Rhinoceros beetle", "Red palm weevil", "Bud rot"],
        "fert_splits": ["1/3rd after monsoon", "2/3rd before post-monsoon"],
        "next_crop": "Intercrops (Banana, Spices)",
    }
}
