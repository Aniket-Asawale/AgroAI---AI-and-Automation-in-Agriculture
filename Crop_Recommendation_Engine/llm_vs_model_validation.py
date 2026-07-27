"""
LLM vs Model Cross-Validation
Compare Claude 4.6 Opus agronomic reasoning against trained RF model.
Runs LIVE predictions through CropRecommender, then compares against
pre-recorded LLM reasoning.

Usage: python Crop_Recommendation_Engine/llm_vs_model_validation.py
"""
import sys
from pathlib import Path

# Ensure parent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.inference import CropRecommender  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "data" / "processed" / "llm_validation_report.txt"


def _write(fh, msg=""):
    """Write a line to file (avoids sys.stdout redirect crash on Windows)."""
    fh.write(msg + "\n")


# ─── Test scenarios: sensor/weather kwargs → CropRecommender.predict() ───
# Each entry: (label, predict_kwargs, llm_expected_crop, llm_reasoning)
SCENARIOS = [
    # -- KHARIF --
    ("1. Soybean: Vidarbha black soil Kharif",
     dict(nitrogen=110, phosphorus=55, potassium=70, temperature=26, moisture=55,
          ec=1400, ph=7.5, weather_temp=28, humidity=72, rainfall=900,
          sunshine=5.5, wind_speed=8, lat=20.5, lon=78.5, altitude=350,
          organic_carbon=0.8, soil_type="Black (Regur)", soil_texture="Clay Loam",
          drainage="Moderate", agro_zone="Vidarbha", season="Kharif", month=7),
     "Soybean",
     "N=110,P=55,K=70 matches soybean profile (N:90-130,P:45-65,K:55-85). Black soil Vidarbha is "
     "THE soybean belt of Maharashtra. Kharif season, temp=26C, rain=900mm -- textbook soybean."),

    ("2. Cotton: Marathwada deep black Kharif",
     dict(nitrogen=170, phosphorus=75, potassium=175, temperature=30, moisture=50,
          ec=1500, ph=7.8, weather_temp=32, humidity=60, rainfall=650,
          sunshine=7.0, wind_speed=10, lat=19.5, lon=76.5, altitude=450,
          organic_carbon=0.7, soil_type="Black (Regur)", soil_texture="Clay",
          drainage="Moderate", agro_zone="Marathwada", season="Kharif", month=8),
     "Cotton",
     "N=170,P=75,K=175 matches cotton (N:140-200,P:60-90,K:150-200). Deep black soil (Regur) with "
     "high K and moderate rainfall=650mm is classic Marathwada cotton belt. Temp=30C optimal."),

    ("3. Rice: Konkan coastal alluvial Kharif",
     dict(nitrogen=175, phosphorus=70, potassium=125, temperature=28, moisture=80,
          ec=1000, ph=6.2, weather_temp=29, humidity=85, rainfall=1800,
          sunshine=4.0, wind_speed=12, lat=17.0, lon=73.3, altitude=20,
          organic_carbon=1.2, soil_type="Alluvial", soil_texture="Silty Clay",
          drainage="Poor", agro_zone="Konkan", season="Kharif", month=7),
     "Rice",
     "Rain=1800mm is the key signal -- only Rice needs 1000-2000mm. Alluvial soil, coastal lat=17, "
     "high humidity=85%. Konkan is Maharashtra's rice bowl."),

    ("4. Sugarcane: W Maharashtra alluvial Kharif",
     dict(nitrogen=250, phosphorus=120, potassium=250, temperature=28, moisture=70,
          ec=1200, ph=7.0, weather_temp=30, humidity=75, rainfall=2000,
          sunshine=6.0, wind_speed=8, lat=16.7, lon=74.2, altitude=550,
          organic_carbon=1.0, soil_type="Alluvial", soil_texture="Clay Loam",
          drainage="Moderate", agro_zone="Western Maharashtra", season="Kharif", month=6),
     "Sugarcane",
     "N=250,P=120,K=250 -- extremely high NPK is the hallmark of sugarcane. "
     "Rain=2000mm, alluvial soil. W Maharashtra (Kolhapur/Sangli) is India's sugar belt."),

    ("5. Groundnut: Sandy red soil Kharif",
     dict(nitrogen=60, phosphorus=55, potassium=110, temperature=29, moisture=35,
          ec=700, ph=6.0, weather_temp=30, humidity=55, rainfall=650,
          sunshine=7.5, wind_speed=9, lat=18.5, lon=75.5, altitude=500,
          organic_carbon=0.5, soil_type="Red", soil_texture="Sandy Loam",
          drainage="Good", agro_zone="Western Maharashtra", season="Kharif", month=7),
     "Groundnut",
     "N=60 is very low (groundnut is a legume, fixes own N). Sandy/red soil with good drainage, "
     "moderate rain=650mm. Classic rainfed groundnut."),

    # -- RABI --
    ("6. Chickpea: Black soil Marathwada Rabi",
     dict(nitrogen=30, phosphorus=50, potassium=100, temperature=18, moisture=30,
          ec=1300, ph=7.5, weather_temp=20, humidity=45, rainfall=350,
          sunshine=8.5, wind_speed=6, lat=19.0, lon=76.0, altitude=500,
          organic_carbon=0.6, soil_type="Black (Regur)", soil_texture="Clay",
          drainage="Moderate", agro_zone="Marathwada", season="Rabi", month=11),
     "Chickpea (Gram)",
     "N=30 (very low -- legume). P=50,K=100 match chickpea. Temp=18C (cool Rabi), "
     "black soil, low rain=350mm. Chickpea is Marathwada's #1 Rabi pulse."),

    ("7. Wheat: Alluvial Indo-Gangetic Rabi (OOD)",
     dict(nitrogen=150, phosphorus=100, potassium=150, temperature=15, moisture=40,
          ec=1100, ph=7.0, weather_temp=18, humidity=50, rainfall=400,
          sunshine=8.0, wind_speed=5, lat=26.85, lon=80.9, altitude=100,
          organic_carbon=0.9, soil_type="Alluvial", soil_texture="Silty Clay",
          drainage="Moderate", agro_zone="Indo-Gangetic", season="Rabi", month=12),
     "Wheat",
     "NPK matches wheat perfectly. BUT lat=26.85 is outside Maharashtra -- OOD expected. "
     "Model should flag low confidence or OOD."),

    ("8. Safflower: Black soil Vidarbha Rabi",
     dict(nitrogen=75, phosphorus=50, potassium=105, temperature=20, moisture=25,
          ec=1400, ph=7.6, weather_temp=22, humidity=35, rainfall=320,
          sunshine=9.0, wind_speed=7, lat=20.0, lon=78.0, altitude=400,
          organic_carbon=0.5, soil_type="Black (Regur)", soil_texture="Clay",
          drainage="Moderate", agro_zone="Vidarbha", season="Rabi", month=12),
     "Safflower",
     "N=75,P=50,K=105 matches safflower. Black soil, dry Rabi (rain=320mm), Vidarbha. "
     "Overlap with Linseed/Sunflower may lower confidence."),

    ("9. Grape: Red soil W Maharashtra Rabi",
     dict(nitrogen=120, phosphorus=100, potassium=215, temperature=22, moisture=35,
          ec=900, ph=6.5, weather_temp=24, humidity=50, rainfall=400,
          sunshine=8.0, wind_speed=6, lat=19.2, lon=74.0, altitude=600,
          organic_carbon=0.7, soil_type="Red", soil_texture="Sandy Loam",
          drainage="Good", agro_zone="Western Maharashtra", season="Rabi", month=1),
     "Grape",
     "K=215 is very high -- grapes demand heavy potassium (K:180-250). Red/laterite soil, "
     "Nashik/Sangli grape belt. Strong match."),

    ("10. Onion: Red sandy N Maharashtra Rabi",
     dict(nitrogen=125, phosphorus=100, potassium=150, temperature=18, moisture=30,
          ec=800, ph=6.5, weather_temp=22, humidity=45, rainfall=400,
          sunshine=8.5, wind_speed=6, lat=20.0, lon=74.0, altitude=550,
          organic_carbon=0.6, soil_type="Red", soil_texture="Sandy Loam",
          drainage="Good", agro_zone="Western Maharashtra", season="Rabi", month=12),
     "Onion",
     "N=125,P=100,K=150 matches onion (N:100-150,P:80-120,K:120-180). Nashik is India's "
     "onion capital. May overlap with Grape due to soil/season similarity."),

    # -- ZAID --
    ("11. Okra: Sandy Laterite Zaid",
     dict(nitrogen=50, phosphorus=25, potassium=40, temperature=36, moisture=20,
          ec=500, ph=6.0, weather_temp=38, humidity=30, rainfall=100,
          sunshine=10.0, wind_speed=10, lat=18.0, lon=75.0, altitude=500,
          organic_carbon=0.4, soil_type="Sandy", soil_texture="Sandy",
          drainage="Excessive", agro_zone="Western Maharashtra", season="Zaid", month=4),
     "Okra",
     "N=50,P=25,K=40 matches okra. Temp=36C (hot Zaid), very low rain=100mm, "
     "sandy/laterite soil. Okra thrives in extreme summer heat."),

    ("12. Chilli: Black soil Zaid",
     dict(nitrogen=70, phosphorus=38, potassium=60, temperature=35, moisture=25,
          ec=1500, ph=7.5, weather_temp=37, humidity=35, rainfall=120,
          sunshine=10.0, wind_speed=8, lat=18.5, lon=76.0, altitude=500,
          organic_carbon=0.5, soil_type="Black (Regur)", soil_texture="Clay",
          drainage="Moderate", agro_zone="Marathwada", season="Zaid", month=4),
     "Chilli",
     "N=70,P=38,K=60 matches chilli. High pH=7.5 on black soil, temp=35C, dry. "
     "Chilli's alkaline soil preference distinguishes it from other Solanaceae."),

    ("13. Brinjal: Alluvial clay Zaid",
     dict(nitrogen=135, phosphorus=80, potassium=140, temperature=27, moisture=45,
          ec=1400, ph=6.7, weather_temp=30, humidity=55, rainfall=380,
          sunshine=8.0, wind_speed=7, lat=18.5, lon=74.5, altitude=500,
          organic_carbon=0.8, soil_type="Alluvial", soil_texture="Clay",
          drainage="Moderate", agro_zone="Western Maharashtra", season="Zaid", month=3),
     "Brinjal",
     "N=135,P=80,K=140 matches brinjal. Alluvial/clay soil, moderate temp=27C. "
     "Brinjal's high NPK demand + clay soil preference is distinctive."),

    # -- EDGE CASES --
    ("14. Ambiguous: mid-range everything Kharif",
     dict(nitrogen=100, phosphorus=60, potassium=120, temperature=25, moisture=45,
          ec=1200, ph=7.0, weather_temp=27, humidity=65, rainfall=600,
          sunshine=6.0, wind_speed=8, lat=19.5, lon=76.0, altitude=450,
          organic_carbon=0.6, soil_type="Black (Regur)", soil_texture="Clay Loam",
          drainage="Moderate", agro_zone="Marathwada", season="Kharif", month=7),
     "Jowar (Kharif) or Soybean",
     "Mid-range NPK overlaps Jowar and Soybean. Jowar is a reasonable default for "
     "generic Kharif conditions on black soil."),

    ("15. Extreme: very high NPK + hot + dry",
     dict(nitrogen=300, phosphorus=150, potassium=300, temperature=40, moisture=15,
          ec=2000, ph=7.5, weather_temp=42, humidity=20, rainfall=50,
          sunshine=11.0, wind_speed=12, lat=19.0, lon=76.0, altitude=400,
          organic_carbon=0.3, soil_type="Black (Regur)", soil_texture="Clay",
          drainage="Moderate", agro_zone="Marathwada", season="Zaid", month=5),
     "No ideal crop (edge case)",
     "N=300,P=150,K=300 + temp=40 + rain=50mm is agronomically extreme. No standard "
     "Maharashtra crop thrives here. Model should show MEDIUM/UNCERTAIN confidence."),
]


def run_validation():
    """Run all scenarios through CropRecommender and compare with LLM reasoning."""
    print("Loading CropRecommender...")
    recommender = CropRecommender()
    print(f"Model loaded: stamp={recommender.model_stamp}, T={recommender.temperature:.3f}")

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        _write(fh, "=" * 100)
        _write(fh, "LLM (Claude 4.6 Opus) vs ML MODEL (Random Forest) -- LIVE Cross-Validation")
        _write(fh, f"Model stamp: {recommender.model_stamp}  |  Temperature: {recommender.temperature:.3f}")
        _write(fh, "=" * 100)
        _write(fh)

        # Header
        header = f"{'#':<4} {'Scenario':<45} {'Model Pred':<18} {'Conf':>6} {'LLM Expects':<22} {'Match?':<8}"
        _write(fh, header)
        _write(fh, "-" * 108)

        agree_count = 0
        disagree_list = []
        results = []

        for label, kwargs, llm_crop, llm_reasoning in SCENARIOS:
            num = label.split(".")[0]
            name = label.split(". ", 1)[1]

            try:
                result = recommender.predict(**kwargs)
                model_crop = result["top_3"][0]["crop"]
                model_conf = result["top_3"][0]["confidence"] * 100
                model_flag = result["confidence_flag"]
                top3_str = ", ".join(
                    f"{r['crop']}({r['confidence']*100:.1f}%)" for r in result["top_3"]
                )
            except Exception as exc:
                model_crop = f"ERROR: {exc}"
                model_conf = 0.0
                model_flag = "ERROR"
                top3_str = str(exc)

            # Check agreement (handle "X or Y" style LLM predictions)
            agree = model_crop in llm_crop or llm_crop in model_crop
            tag = "YES" if agree else "DISAGREE"
            if agree:
                agree_count += 1
            else:
                disagree_list.append((label, model_crop, model_conf, model_flag,
                                      llm_crop, llm_reasoning, top3_str))

            results.append((label, model_crop, model_conf, model_flag,
                            llm_crop, llm_reasoning, top3_str))

            _write(fh, f"{num:<4} {name:<45} {model_crop:<18} {model_conf:>5.1f}% {llm_crop:<22} {tag:<8}")

        _write(fh)
        _write(fh, f"Agreement: {agree_count}/{len(SCENARIOS)} "
                    f"({agree_count/len(SCENARIOS)*100:.0f}%)")

        # Disagreement details
        if disagree_list:
            _write(fh)
            _write(fh, "=" * 100)
            _write(fh, f"DISAGREEMENTS ({len(disagree_list)} cases)")
            _write(fh, "=" * 100)
            for lbl, mc, mconf, mflag, lc, lreason, t3 in disagree_list:
                _write(fh, f"\n  {lbl}")
                _write(fh, f"    Model says: {mc} ({mconf:.1f}%) [{mflag}]")
                _write(fh, f"    Top-3:      {t3}")
                _write(fh, f"    LLM says:   {lc}")
                _write(fh, f"    Reasoning:  {lreason}")
                if mconf < 60:
                    _write(fh, f"    --> Model was UNCERTAIN ({mconf:.1f}% < 60%)")
                elif mconf < 75:
                    _write(fh, f"    --> Model had MEDIUM confidence ({mconf:.1f}%)")

        # Full reasoning
        _write(fh)
        _write(fh, "=" * 100)
        _write(fh, "DETAILED LLM REASONING + LIVE MODEL OUTPUT")
        _write(fh, "=" * 100)
        for lbl, mc, mconf, mflag, lc, lreason, t3 in results:
            agree = mc in lc or lc in mc
            tag = "AGREE" if agree else "DISAGREE"
            _write(fh, f"\n  {lbl}")
            _write(fh, f"    Model: {mc} ({mconf:.1f}%, {mflag})  |  LLM: {lc}  |  [{tag}]")
            _write(fh, f"    Top-3: {t3}")
            _write(fh, f"    LLM:   {lreason}")

        # Resolution strategy
        _write(fh)
        _write(fh, "=" * 100)
        _write(fh, "RESOLUTION STRATEGY")
        _write(fh, "=" * 100)
        _write(fh, "  HIGH (>=75%)         -> Trust the model.")
        _write(fh, "  MEDIUM (60-75%)      -> Show top-3 + LLM alternative. Let farmer choose.")
        _write(fh, "  UNCERTAIN (<60%)     -> Show top-3 + LLM + 'Consult local KVK'.")
        _write(fh, "  OUT-OF-DISTRIBUTION  -> Reject: 'Model trained for Maharashtra only.'")
        _write(fh)
        _write(fh, "=" * 100)
        _write(fh, "VALIDATION COMPLETE")
        _write(fh, "=" * 100)

    print(f"\nResults written to {OUT_FILE}")
    print(f"Agreement: {agree_count}/{len(SCENARIOS)} ({agree_count/len(SCENARIOS)*100:.0f}%)")
    if disagree_list:
        print(f"Disagreements: {len(disagree_list)}")
        for lbl, mc, mconf, mflag, lc, lreason, t3 in disagree_list:
            print(f"  {lbl}: Model={mc}({mconf:.1f}%) vs LLM={lc}")


if __name__ == "__main__":
    run_validation()
