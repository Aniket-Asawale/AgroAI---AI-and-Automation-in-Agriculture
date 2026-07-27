"""
API Gateway routes for Crop Recommendation Engine (port 8001).
Proxies prediction, rotation, amendment, and weather endpoints.
"""

import httpx
from fastapi import APIRouter, HTTPException, Query, Depends, Request, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from config import settings

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])

BASE = settings.CROP_RECOMMENDATION_URL
TIMEOUT = settings.REQUEST_TIMEOUT


async def _get_client():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client

# ─── Schemas for Swagger UI ───
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
    prev_crop: Optional[str] = Field(None, description="Previous crop")
    irrigation_type: Optional[str] = Field(None, description="'Rainfed','Drip','Sprinkler','Flood'")
    irrigation_available: Optional[int] = Field(0, description="0=rainfed, 1=irrigated")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nitrogen": 45.5,
                "phosphorus": 20.0,
                "potassium": 35.0,
                "temperature": 26.5,
                "moisture": 65.0,
                "ec": 300.0,
                "ph": 6.8,
                "weather_temp": 28.5,
                "humidity": 60.0,
                "rainfall": 800.0,
                "sunshine": 8.5,
                "wind_speed": 12.0,
                "lat": 18.5204,
                "lon": 73.8567,
                "altitude": 560.0,
                "organic_carbon": 1.2,
                "soil_type": "Black (Regur)",
                "soil_texture": "Clay Loam",
                "drainage": "Moderate",
                "agro_zone": "Western Maharashtra",
                "season": "Kharif",
                "month": 6,
                "prev_crop": "Wheat",
                "irrigation_type": "Drip",
                "irrigation_available": 1
            }
        }
    }

class LivePredictRequest(BaseModel):
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "nitrogen": 45.5,
                "phosphorus": 20.0,
                "potassium": 35.0,
                "temperature": 26.5,
                "moisture": 65.0,
                "ec": 300.0,
                "ph": 6.8,
                "lat": 18.5204,
                "lon": 73.8567,
                "altitude": 560.0,
                "organic_carbon": 1.2,
                "soil_type": "Black (Regur)",
                "soil_texture": "Clay Loam",
                "drainage": "Moderate",
                "agro_zone": "Western Maharashtra",
                "season": "Kharif",
                "month": 6,
                "prev_crop": "Wheat",
                "irrigation_type": "Drip",
                "irrigation_available": 1
            }
        }
    }

class RotationRequest(BaseModel):
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "nitrogen": 45.5,
                "phosphorus": 20.0,
                "potassium": 35.0,
                "temperature": 26.5,
                "moisture": 65.0,
                "ec": 300.0,
                "ph": 6.8,
                "lat": 18.5204,
                "lon": 73.8567,
                "altitude": 560.0,
                "organic_carbon": 1.2,
                "soil_type": "Black (Regur)",
                "soil_texture": "Clay Loam",
                "drainage": "Moderate",
                "agro_zone": "Western Maharashtra"
            }
        }
    }

class AmendmentRequest(BaseModel):
    crop_name: str = Field(..., description="Target crop name")
    nitrogen: float = Field(..., ge=0, le=500, description="Current soil N")
    phosphorus: float = Field(..., ge=0, le=300, description="Current soil P")
    potassium: float = Field(..., ge=0, le=500, description="Current soil K")
    field_area_ha: float = Field(1.0, gt=0, description="Field area in ha")

    model_config = {
        "json_schema_extra": {
            "example": {
                "crop_name": "Sugarcane",
                "nitrogen": 45.5,
                "phosphorus": 20.0,
                "potassium": 35.0,
                "field_area_ha": 2.5
            }
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_crop": "Cotton",
                "nitrogen": 80,
                "phosphorus": 40,
                "potassium": 60,
                "ph": 7.2,
                "ec": 1800,
                "soil_type": "Black (Regur)",
                "drainage": "Moderate",
                "rainfall": 650,
                "weather_temp": 30,
                "agro_zone": "Vidarbha",
                "field_area_ha": 2.0
            }
        }
    }


@router.get("/health")
async def crop_health(client: httpx.AsyncClient = Depends(_get_client)):
    """Crop Recommendation Engine health check."""
    try:
        r = await client.get(f"{BASE}/health")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.get("/crops")
async def list_crops(client: httpx.AsyncClient = Depends(_get_client)):
    """List all supported crops."""
    try:
        r = await client.get(f"{BASE}/crops")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/predict")
async def predict(payload: PredictRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Get crop recommendation from full inputs."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/predict", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/predict/live")
async def predict_live(payload: LivePredictRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Predict with auto-fetched weather from Open-Meteo."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/predict/live", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/rotation")
async def rotation(payload: RotationRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Generate full-year crop rotation plan."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/rotation", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/amendments")
async def amendments(payload: AmendmentRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Calculate fertilizer amendments for a target crop."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/amendments", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/reverse")
async def reverse_recommendation(payload: ReverseRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Evaluate a user-chosen target crop — returns feasibility, deficits, fixes, and yield tips."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/reverse", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.post("/decide")
async def evaluate_decision(payload: ReverseRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    """Evaluate a user-chosen target crop — returns full decision report, yield, financials, action plan."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.post(f"{BASE}/decide", content=body, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.get("/weather")
async def crop_weather(
    lat: float = Query(...),
    lon: float = Query(...),
    client: httpx.AsyncClient = Depends(_get_client),
):
    """Fetch live weather for a location (Open-Meteo via Crop Engine)."""
    try:
        r = await client.get(f"{BASE}/weather", params={"lat": lat, "lon": lon})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Crop Recommendation unreachable: {exc}")


@router.get("/dashboard/{path:path}")
@router.post("/dashboard/{path:path}")
@router.put("/dashboard/{path:path}")
@router.delete("/dashboard/{path:path}")
@router.patch("/dashboard/{path:path}")
@router.options("/dashboard/{path:path}")
@router.head("/dashboard/{path:path}")
async def crop_dashboard_proxy(
    path: str,
    request: Request,
    client: httpx.AsyncClient = Depends(_get_client)
):
    """Proxy all requests to Streamlit dashboard on port 8501."""
    try:
        # Get query params
        query_string = str(request.url.query) if request.url.query else ""
        
        # Build the target URL - Streamlit dashboard runs on port 8501
        target_url = f"http://127.0.0.1:8501/{path}"
        if query_string:
            target_url += f"?{query_string}"
        
        # Get request body if present
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
            except:
                body = None
        
        # Forward the request with the same method
        r = await client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ["host", "connection"]},
            follow_redirects=True
        )
        
        return StreamingResponse(
            iter([r.content]),
            status_code=r.status_code,
            headers=dict(r.headers),
            media_type=r.headers.get("content-type", "application/octet-stream")
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Dashboard unreachable: {exc}")

