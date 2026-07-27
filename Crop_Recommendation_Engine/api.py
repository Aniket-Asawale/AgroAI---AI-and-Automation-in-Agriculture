"""
Standalone FastAPI for the Crop Recommendation Engine.

Endpoints:
    POST /predict            — Single crop recommendation (full inputs)
    POST /predict/live       — Prediction with auto-fetched weather
    POST /rotation           — Full-year rotation plan (Kharif→Rabi→Zaid)
    POST /amendments         — Fertilizer amendment calculator
    POST /reverse            — Reverse recommendation: user picks a target
                               crop; engine reports feasibility, deficits,
                               fixes, and yield tips.
    GET  /weather            — Fetch live weather for a lat/lon
    GET  /health             — Health check + model info
    GET  /crops              — List all supported crops & profiles

Usage:
    cd Crop_Recommendation_Engine
    uvicorn api:app --reload --port 8001
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.inference import CropRecommender  # noqa: E402
from generators.crop_profiles import ALL_CROPS, CROP_TO_FAMILY  # noqa: E402
from config import (  # noqa: E402
    CONFIDENCE_HIGH, CONFIDENCE_LOW, MH_LAT_RANGE, MH_LON_RANGE,
    VALID_SEASONS, VALID_SOIL_TYPES,
)

# ─── App setup ───
app = FastAPI(
    title="Crop Recommendation Engine API",
    description="ML-powered crop recommendation for Maharashtra, India. "
                "Uses a calibrated Random Forest with 40 agronomic features.",
    version="1.0.0",
)

# Load model once at startup
recommender: Optional[CropRecommender] = None


@app.on_event("startup")
def load_model():
    global recommender
    recommender = CropRecommender()


# ─── Request / Response schemas ───

class PredictRequest(BaseModel):
    nitrogen: float = Field(..., ge=0, le=500, description="Soil N (mg/kg)")
    phosphorus: float = Field(..., ge=0, le=300, description="Soil P (mg/kg)")
    potassium: float = Field(..., ge=0, le=500, description="Soil K (mg/kg)")
    temperature: float = Field(..., ge=0, le=55, description="Soil temp (°C)")
    moisture: float = Field(..., ge=0, le=100, description="Soil moisture (%RH)")
    ec: float = Field(..., ge=0, le=20000, description="Electrical conductivity (μS/cm)")
    ph: float = Field(..., ge=3.0, le=10.0, description="Soil pH")
    weather_temp: float = Field(..., ge=-5, le=55, description="Air temp (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    rainfall: float = Field(..., ge=0, le=5000, description="Seasonal rainfall (mm)")
    sunshine: float = Field(..., ge=0, le=14, description="Sunshine hrs/day")
    wind_speed: float = Field(..., ge=0, le=100, description="Wind speed (km/h)")
    lat: float = Field(..., ge=5, le=40, description="Latitude")
    lon: float = Field(..., ge=65, le=100, description="Longitude")
    altitude: float = Field(..., ge=0, le=5000, description="Altitude (m)")
    organic_carbon: float = Field(..., ge=0, le=10, description="Organic carbon (%)")
    soil_type: str = Field(..., description="Soil type, e.g. 'Black (Regur)', 'Red'")
    soil_texture: str = Field(..., description="e.g. 'Clay Loam', 'Sandy Loam'")
    drainage: str = Field(..., description="e.g. 'Moderate', 'Good', 'Poor'")
    agro_zone: str = Field(..., description="e.g. 'Vidarbha', 'Marathwada'")
    season: str = Field(..., description="'Kharif', 'Rabi', or 'Zaid'")
    month: int = Field(..., ge=1, le=12, description="Month number")
    prev_crop: Optional[str] = Field(None, description="Previous crop for rotation adjustment")
    irrigation_type: Optional[str] = Field(None, description="'Rainfed','Drip','Sprinkler','Flood'")
    irrigation_available: Optional[int] = Field(0, ge=0, le=1, description="0=rainfed, 1=irrigated")
    llm_verify: Optional[bool] = Field(None, description="Run LLM second-opinion (auto if key present)")


class LivePredictRequest(BaseModel):
    """Prediction with auto-fetched weather. Only soil + location needed."""
    nitrogen: float = Field(..., ge=0, le=500)
    phosphorus: float = Field(..., ge=0, le=300)
    potassium: float = Field(..., ge=0, le=500)
    temperature: float = Field(..., ge=0, le=55, description="Soil temp (°C)")
    moisture: float = Field(..., ge=0, le=100)
    ec: float = Field(..., ge=0, le=20000)
    ph: float = Field(..., ge=3.0, le=10.0)
    lat: float = Field(..., ge=5, le=40)
    lon: float = Field(..., ge=65, le=100)
    altitude: float = Field(..., ge=0, le=5000)
    organic_carbon: float = Field(..., ge=0, le=10)
    soil_type: str
    soil_texture: str
    drainage: str
    agro_zone: str
    season: str
    month: int = Field(..., ge=1, le=12)
    prev_crop: Optional[str] = None
    irrigation_type: Optional[str] = None
    irrigation_available: Optional[int] = 0
    # Weather fields optional — auto-filled from Open-Meteo if missing
    weather_temp: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    sunshine: Optional[float] = None
    wind_speed: Optional[float] = None
    llm_verify: Optional[bool] = Field(None, description="Run LLM second-opinion (auto if key present)")


class RotationRequest(BaseModel):
    """Full-year rotation plan. Season/weather auto-filled per season."""
    nitrogen: float = Field(..., ge=0, le=500)
    phosphorus: float = Field(..., ge=0, le=300)
    potassium: float = Field(..., ge=0, le=500)
    temperature: float = Field(..., ge=0, le=55)
    moisture: float = Field(..., ge=0, le=100)
    ec: float = Field(..., ge=0, le=20000)
    ph: float = Field(..., ge=3.0, le=10.0)
    lat: float = Field(..., ge=5, le=40)
    lon: float = Field(..., ge=65, le=100)
    altitude: float = Field(..., ge=0, le=5000)
    organic_carbon: float = Field(..., ge=0, le=10)
    soil_type: str
    soil_texture: str
    drainage: str
    agro_zone: str


class AmendmentRequest(BaseModel):
    crop_name: str = Field(..., description="Target crop name")
    nitrogen: float = Field(..., ge=0, le=500, description="Current soil N (mg/kg)")
    phosphorus: float = Field(..., ge=0, le=300, description="Current soil P (mg/kg)")
    potassium: float = Field(..., ge=0, le=500, description="Current soil K (mg/kg)")
    field_area_ha: float = Field(1.0, gt=0, description="Field area in hectares")


class ReverseRequest(BaseModel):
    """Reverse recommendation — evaluate a user-chosen target crop."""
    target_crop: str = Field(..., description="Target crop the farmer wants to grow")
    nitrogen: float = Field(..., ge=0, le=500)
    phosphorus: float = Field(..., ge=0, le=300)
    potassium: float = Field(..., ge=0, le=500)
    ph: float = Field(..., ge=3.0, le=10.0)
    ec: float = Field(..., ge=0, le=20000)
    soil_type: str
    drainage: str
    rainfall: float = Field(..., ge=0, le=5000, description="Seasonal rainfall (mm)")
    weather_temp: float = Field(..., ge=-5, le=55, description="Ambient air temperature (°C)")
    agro_zone: Optional[str] = Field("", description="e.g. 'Vidarbha'")
    field_area_ha: float = Field(1.0, gt=0)




# ─── Endpoints ───

from fastapi.responses import RedirectResponse

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Health check — returns model info and supported values."""
    return {
        "status": "healthy",
        "model_stamp": recommender.model_stamp,
        "temperature": recommender.temperature,
        "num_classes": len(recommender.crop_labels),
        "confidence_thresholds": {
            "HIGH": f">= {CONFIDENCE_HIGH}",
            "LOW": f"< {CONFIDENCE_LOW}",
        },
        "maharashtra_bounds": {
            "lat": list(MH_LAT_RANGE),
            "lon": list(MH_LON_RANGE),
        },
        "valid_seasons": sorted(VALID_SEASONS),
        "valid_soil_types": sorted(VALID_SOIL_TYPES),
    }


@app.get("/crops")
def list_crops():
    """List all supported crops with their season and family."""
    crops = []
    for season, season_crops in ALL_CROPS.items():
        for name in season_crops:
            crops.append({
                "name": name,
                "season": season,
                "family": CROP_TO_FAMILY.get(name, "Unknown"),
            })
    return {"crops": crops, "total": len(crops)}


def _json_response(data: dict) -> JSONResponse:
    """Return JSONResponse with numpy-safe encoding."""
    content = json.loads(json.dumps(data, cls=NumpyEncoder))
    return JSONResponse(content=content)


@app.post("/predict")
def predict(req: PredictRequest):
    """Generate top-3 crop recommendation with confidence flags."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = recommender.predict(**kwargs)
        return _json_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/live")
def predict_live(req: LivePredictRequest):
    """Predict with auto-fetched weather from Open-Meteo."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = recommender.predict_with_live_weather(**kwargs)
        return _json_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/rotation")
def rotation_plan(req: RotationRequest):
    """Generate full-year crop rotation plan: Kharif → Rabi → Zaid."""
    try:
        kwargs = req.model_dump(exclude_none=True)
        result = recommender.plan_rotation(**kwargs)
        return _json_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rotation planning failed: {exc}")


@app.post("/amendments")
def amendments(req: AmendmentRequest):
    """Calculate fertilizer amendments for a target crop."""
    try:
        result = CropRecommender.calculate_amendments(
            crop_name=req.crop_name,
            nitrogen=req.nitrogen,
            phosphorus=req.phosphorus,
            potassium=req.potassium,
            field_area_ha=req.field_area_ha,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return _json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Amendment calc failed: {exc}")


@app.post("/reverse")
def reverse_recommendation(req: ReverseRequest):
    """Evaluate a user-chosen target crop against current field conditions.

    Returns feasibility score, NPK/pH/EC/rainfall/temperature deficits,
    specific actionable fixes, and yield-maximisation tips — exactly the
    questions the farmer might otherwise ask an online LLM.
    """
    try:
        result = CropRecommender.calculate_reverse_recommendation(
            target_crop=req.target_crop,
            nitrogen=req.nitrogen,
            phosphorus=req.phosphorus,
            potassium=req.potassium,
            ph=req.ph,
            ec=req.ec,
            soil_type=req.soil_type,
            drainage=req.drainage,
            rainfall=req.rainfall,
            weather_temp=req.weather_temp,
            agro_zone=req.agro_zone or "",
            field_area_ha=req.field_area_ha,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return _json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reverse recommendation failed: {exc}")


@app.post("/decide")
def evaluate_decision(req: ReverseRequest):
    """Evaluate a user-chosen target crop decision.

    Returns a full decision report including feasibility score, yield estimate,
    financial projections, and a 4-phase action plan.
    """
    try:
        result = CropRecommender.evaluate_crop_decision(
            target_crop=req.target_crop,
            nitrogen=req.nitrogen,
            phosphorus=req.phosphorus,
            potassium=req.potassium,
            ph=req.ph,
            ec=req.ec,
            soil_type=req.soil_type,
            drainage=req.drainage,
            rainfall=req.rainfall,
            weather_temp=req.weather_temp,
            agro_zone=req.agro_zone or "",
            field_area_ha=req.field_area_ha,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return _json_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Crop decision evaluation failed: {exc}")


@app.get("/weather")
def get_weather(lat: float, lon: float):
    """Fetch live weather for a location from Open-Meteo."""
    try:
        result = CropRecommender.fetch_weather(lat, lon)
        return {"lat": lat, "lon": lon, "weather": result}
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Weather fetch failed: {exc}")