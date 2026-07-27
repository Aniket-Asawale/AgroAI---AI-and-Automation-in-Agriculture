"""
Public API routes - NO API key required.
Status pages, health checks, and dashboards accessible to all.
"""

import logging
from typing import Dict, Any

import httpx
from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Public Status"])


@router.get("/sensor/health", tags=["AgroSensor"])
async def sensor_health():
    """Public health check for AgroSensor service - NO API key required.
    
    Returns simple status of the sensor service.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.AGROSENSOR_URL}/health")
            return response.json()
    except Exception as exc:
        logger.error(f"Sensor health check failed: {exc}")
        return {
            "status": "down",
            "error": str(exc),
            "service": "AgroSensor",
            "port": 8000,
            "message": "AgroSensor service is not reachable. Start it with: AgroManager [3] → [2]"
        }


@router.get("/sensor/dashboard", tags=["AgroSensor"])
async def sensor_dashboard():
    """Sensor service status dashboard - NO API key required.
    
    Shows detailed information about the sensor service including connection status,
    available endpoints, and troubleshooting information.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            health = await client.get(f"{settings.AGROSENSOR_URL}/health")
            health_data = health.json()
            
        return {
            "status": "operational",
            "service": "AgroSensor (Soil Sensor & Weather Data)",
            "port": 8000,
            "health": health_data,
            "available_endpoints": [
                "GET /api/sensor/locations - Get all sensor locations",
                "GET /api/sensor/weather/{location} - Get weather for location",
                "GET /api/sensor/data/{location} - Get latest sensor data",
            ],
            "access": {
                "public_url": "https://agroaiapp.me/sensor/health",
                "local_url": "http://127.0.0.1:8000/health",
                "dashboard": "https://agroaiapp.me/dashboard"
            }
        }
    except Exception as exc:
        logger.error(f"Sensor dashboard check failed: {exc}")
        return {
            "status": "offline",
            "service": "AgroSensor",
            "port": 8000,
            "error": str(exc),
            "troubleshooting": {
                "step_1": "Open AgroManager.bat",
                "step_2": "Select [3] Start Specific Service",
                "step_3": "Select [2] AgroSensor",
                "step_4": "Wait for window to appear",
                "step_5": "Refresh this page"
            },
            "manual_start": "cd AgroSensor && venv\\Scripts\\python.exe main.py"
        }


@router.get("/crop/health", tags=["Crop Recommendation"])
async def crop_health():
    """Public health check for Crop API - NO API key required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.CROP_RECOMMENDATION_URL}/health")
            return response.json()
    except Exception as exc:
        logger.error(f"Crop health check failed: {exc}")
        return {
            "status": "down",
            "error": str(exc),
            "service": "Crop Recommendation Engine",
            "port": 8001
        }


@router.get("/disease/health", tags=["Disease Detection"])
async def disease_health():
    """Public health check for Disease Detection API - NO API key required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.PLANT_DISEASE_URL}/health")
            return response.json()
    except Exception as exc:
        logger.error(f"Disease health check failed: {exc}")
        return {
            "status": "down",
            "error": str(exc),
            "service": "Plant Disease Detection",
            "port": 8003
        }


@router.get("/auth/health", tags=["Auth"])
async def auth_health():
    """Public health check for Auth service - NO API key required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.AUTH_URL}/health")
            return response.json()
    except Exception as exc:
        logger.error(f"Auth health check failed: {exc}")
        return {
            "status": "down",
            "error": str(exc),
            "service": "Auth Service",
            "port": 8002
        }
