"""
API Gateway routes for AgroSensor service (port 8000).
Proxies sensor, weather, and location endpoints.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse

from config import settings

router = APIRouter(prefix="/sensor", tags=["AgroSensor"])

BASE = settings.AGROSENSOR_URL
TIMEOUT = settings.REQUEST_TIMEOUT


async def _get_client():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


@router.get("/health")
async def sensor_health(client: httpx.AsyncClient = Depends(_get_client)):
    """AgroSensor health check."""
    try:
        r = await client.get(f"{BASE}/api/health")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/live")
async def sensor_live(
    city: Optional[str] = Query(None),
    client: httpx.AsyncClient = Depends(_get_client),
):
    """Get latest sensor reading."""
    try:
        params = {"city": city} if city else {}
        r = await client.get(f"{BASE}/api/sensor/live", params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/history")
async def sensor_history(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=5000),
    client: httpx.AsyncClient = Depends(_get_client),
):
    """Paginated sensor history."""
    try:
        params = {"page": page, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if city:
            params["city"] = city
        r = await client.get(f"{BASE}/api/sensor/history", params=params)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.post("/read")
async def sensor_read(client: httpx.AsyncClient = Depends(_get_client)):
    """Trigger an immediate sensor read."""
    try:
        r = await client.post(f"{BASE}/api/sensor/read")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/metadata")
async def sensor_metadata(
    city: Optional[str] = Query(None),
    client: httpx.AsyncClient = Depends(_get_client),
):
    """Get sensor metadata."""
    try:
        params = {"city": city} if city else {}
        r = await client.get(f"{BASE}/api/sensor/metadata", params=params)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/weather")
async def weather(client: httpx.AsyncClient = Depends(_get_client)):
    """Get current weather from AgroSensor's Open-Meteo integration."""
    try:
        r = await client.get(f"{BASE}/api/weather")
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/weather/cities")
async def weather_cities(client: httpx.AsyncClient = Depends(_get_client)):
    """List all available cities."""
    try:
        r = await client.get(f"{BASE}/api/weather/cities")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")


@router.get("/locations")
async def list_locations(client: httpx.AsyncClient = Depends(_get_client)):
    """List active locations."""
    try:
        r = await client.get(f"{BASE}/api/locations")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AgroSensor unreachable: {exc}")




