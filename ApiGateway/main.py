"""
AgroModules — API Gateway

Public entrypoint (via Cloudflare Tunnel) that routes to local microservices.

Routes:
  - /docs, /redoc, /openapi.json  → FastAPI docs
  - /health                       → gateway + backend reachability
  - /api/auth/*                   → Auth service (8002)   [API key optional/required via config]
  - /api/sensor/*                 → AgroSensor (8000)     [API key optional/required via config]
  - /api/crop/*                   → Crop API (8001)       [API key optional/required via config]
  - /api/disease/*                → Disease API (8003)    [API key optional/required via config]
  - /dashboard/*, /crop-recommendation/dashboard/*  → Streamlit dashboard (8501) [NO API key]
  - /sensor/dashboard/*           → AgroSensor API (8000) [NO API key]
  - /plant-disease/dashboard/*    → Disease API (8003) [NO API key]
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketException, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse

from config import settings
from routes.auth import router as auth_router
from routes.sensor import router as sensor_router
from routes.crop import router as crop_router
from routes.disease import router as disease_router

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: log backend service URLs (don't block on checks)."""
    services = {
        "Auth": settings.AUTH_URL,
        "AgroSensor": settings.AGROSENSOR_URL,
        "Crop Recommendation": settings.CROP_RECOMMENDATION_URL,
        "Plant Disease": settings.PLANT_DISEASE_URL,
    }
    logger.info("=" * 60)
    logger.info("AGROMODULES API GATEWAY STARTING")
    logger.info("=" * 60)
    logger.info("Backend services configured at:")
    for name, url in services.items():
        logger.info("  %s → %s", name, url)
    logger.info("=" * 60)
    logger.info("Gateway listening on http://127.0.0.1:8080")
    logger.info("Public URL: https://agroaiapp.me (via Cloudflare Tunnel)")
    logger.info("Docs: http://127.0.0.1:8080/docs")
    logger.info("=" * 60)

    yield  # App is running
    logger.info("API Gateway shutting down")


app = FastAPI(
    title="AgroModules API Gateway",
    description=(
        "Unified API gateway for the AgroModules platform.\n\n"
        "**Services:**\n"
        "- `/api/auth/*` — User authentication, registration, diagnosis history\n"
        "- `/api/sensor/*` — Soil sensor data, weather, locations (AgroSensor)\n"
        "- `/api/crop/*` — ML crop recommendation, rotation, amendments\n"
        "- `/api/disease/*` — Plant disease detection from leaf images\n\n"
        "Designed as the single entry point for AgroMobile and other client apps."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow mobile & web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Depends, Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    if not settings.REQUIRE_API_KEY:
        return None
    valid_keys = [k.strip() for k in settings.VALID_API_KEYS.split(",") if k.strip()]
    if api_key in valid_keys:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API key"
    )

api_deps = [Depends(get_api_key)] if settings.REQUIRE_API_KEY else []

# Mount service routers under /api securely
app.include_router(auth_router, prefix="/api", dependencies=api_deps)
app.include_router(sensor_router, prefix="/api", dependencies=api_deps)
app.include_router(crop_router, prefix="/api", dependencies=api_deps)
app.include_router(disease_router, prefix="/api", dependencies=api_deps)





from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api-docs", include_in_schema=False)
def api_docs_redirect():
    """Alias for /docs (public domain compatibility)."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def gateway_health():
    """Gateway health + backend service status."""
    services = {
        "auth": settings.AUTH_URL,
        "agrosensor": settings.AGROSENSOR_URL,
        "crop_recommendation": settings.CROP_RECOMMENDATION_URL,
        "plant_disease": settings.PLANT_DISEASE_URL,
    }
    statuses = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in services.items():
            try:
                r = await client.get(f"{url}/health")
                statuses[name] = {"status": "up", "code": r.status_code}
            except Exception:
                statuses[name] = {"status": "down"}

    all_up = all(s["status"] == "up" for s in statuses.values())
    return {
        "gateway": "healthy",
        "services": statuses,
        "all_services_up": all_up,
    }


@app.get("/dashboard", include_in_schema=False)
async def dashboard_redirect():
    """Redirect /dashboard to Streamlit Cloud crop dashboard."""
    return RedirectResponse(url="https://croprecommendationengine.streamlit.app")


@app.get("/dashboard/", include_in_schema=False)
async def dashboard_redirect_root():
    """Redirect /dashboard/ to Streamlit Cloud crop dashboard."""
    return RedirectResponse(url="https://croprecommendationengine.streamlit.app")


@app.get("/status")
async def public_status():
    """System status for public domain (https://agroaiapp.me/status)."""
    services = {
        "auth": settings.AUTH_URL,
        "agrosensor": settings.AGROSENSOR_URL,
        "crop_recommendation": settings.CROP_RECOMMENDATION_URL,
        "plant_disease": settings.PLANT_DISEASE_URL,
    }
    statuses = {}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in services.items():
            try:
                r = await client.get(f"{url}/health")
                statuses[name] = {"status": "up", "code": r.status_code}
            except Exception:
                statuses[name] = {"status": "down"}
    
    return {
        "gateway": "healthy",
        "public_domain": "https://agroaiapp.me",
        "services": statuses,
        "endpoints": {
            "api_docs": "https://agroaiapp.me/api-docs",
            "crop_dashboard": "https://agroaiapp.me/crop-recommendation/dashboard",
            "sensor_dashboard": "https://agroaiapp.me/sensor/dashboard",
            "disease_dashboard": "https://agroaiapp.me/plant-disease/dashboard",
        }
    }


@app.get("/crop-recommendation/dashboard/{path:path}", include_in_schema=False)
async def redirect_crop_dashboard(path: str):
    """Redirect crop dashboard to Streamlit Cloud (no longer proxied locally)."""
    return RedirectResponse(url="https://croprecommendationengine.streamlit.app")


@app.api_route("/sensor/static/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_sensor_static(request: Request, path: str):
    """Proxy Sensor Dashboard static files to local dashboard directory (port 8502)."""
    from fastapi.responses import StreamingResponse

    # Build the upstream URL - Static files are served from the dashboard directory on port 8502
    upstream_url = f"http://127.0.0.1:8502/{path}"

    # Prepare headers, excluding hop-by-hop headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Let the target server set its own Host header
    headers.pop("connection", None)
    headers.pop("upgrade", None)
    headers.pop("sec-websocket-key", None)
    headers.pop("sec-websocket-version", None)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            # Make the upstream request
            response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers
            )

            # Create response, handling potential issues with headers
            response_headers = dict(response.headers)
            # Remove problematic headers
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("connection", None)

            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=response_headers
            )
        except httpx.ConnectError:
            logger.error(f"Cannot connect to sensor dashboard static files on port 8502")
            return {"error": "Sensor dashboard static files unavailable", "details": "Connection refused - sensor dashboard may not be running"}, 502
        except Exception as e:
            logger.error(f"Sensor dashboard static file proxy error: {e}")
            return {"error": "Sensor dashboard static files unavailable", "details": str(e)}, 502


@app.api_route("/sensor/dashboard/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_sensor_dashboard(request: Request, path: str):
    """Proxy Sensor Dashboard requests to AgroSensor API (port 8000)."""
    from fastapi.responses import StreamingResponse

    # Build the upstream URL - AgroSensor static dashboard runs on 8502
    # When path is empty, serve index.html
    actual_path = path if path and path != '/' else 'index.html'
    upstream_url = f"http://127.0.0.1:8502/{actual_path}"

    # Prepare headers, excluding hop-by-hop headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Let the target server set its own Host header
    headers.pop("connection", None)
    headers.pop("upgrade", None)
    headers.pop("sec-websocket-key", None)
    headers.pop("sec-websocket-version", None)

    # Determine the method
    method = request.method

    # Get the request body if present
    body = await request.body() if method in ["POST", "PUT", "PATCH", "DELETE"] else b""

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            # Make the upstream request
            response = await client.request(
                method=method,
                url=upstream_url,
                headers=headers,
                content=body
            )

            # Create response, handling potential issues with headers
            response_headers = dict(response.headers)
            # Remove problematic headers
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("connection", None)

            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=response_headers
            )
        except httpx.ConnectError:
            logger.error(f"Cannot connect to sensor dashboard on port 8502")
            return {"error": "Sensor dashboard unavailable", "details": "Connection refused - sensor dashboard may not be running"}, 502
        except Exception as e:
            logger.error(f"Sensor dashboard proxy error: {e}")
            return {"error": "Sensor dashboard unavailable", "details": str(e)}, 502


# Remove the sensor dashboard WebSocket proxy since AgroSensor doesn't use WebSockets
# The sensor dashboard is actually served by the AgroSensor API on port 8000


@app.api_route("/plant-disease/dashboard/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_disease_dashboard(request: Request, path: str):
    """Proxy Plant Disease Dashboard requests to Gradio (port 8003)."""
    from fastapi.responses import StreamingResponse
    
    # Build the upstream URL - Plant Disease UI runs on port 7860
    upstream_url = f"http://127.0.0.1:7860/{path}"
    
    # Prepare headers, excluding hop-by-hop headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Let the target server set its own Host header
    headers.pop("connection", None)
    headers.pop("upgrade", None)
    headers.pop("sec-websocket-key", None)
    headers.pop("sec-websocket-version", None)
    
    # Determine the method
    method = request.method
    
    # Get the request body if present
    body = await request.body() if method in ["POST", "PUT", "PATCH", "DELETE"] else b""
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            # Make the upstream request
            response = await client.request(
                method=method,
                url=upstream_url,
                headers=headers,
                content=body
            )
            
            # Create response, handling potential issues with headers
            response_headers = dict(response.headers)
            # Remove problematic headers
            response_headers.pop("content-length", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("connection", None)
            
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=response_headers
            )
        except httpx.ConnectError:
            logger.error(f"Cannot connect to disease dashboard on port 8003")
            return {"error": "Disease dashboard unavailable", "details": "Connection refused - disease service may not be running"}, 502
        except Exception as e:
            logger.error(f"Disease dashboard proxy error: {e}")
            return {"error": "Disease dashboard unavailable", "details": str(e)}, 502


@app.websocket("/plant-disease/dashboard/{path:path}")
async def websocket_disease_dashboard(websocket: WebSocket, path: str):
    """WebSocket proxy for Plant Disease Dashboard (Gradio)."""
    await websocket.accept()
    upstream_url = f"ws://127.0.0.1:7860/{path}"  # Plant disease UI runs on port 7860
    try:
        async with websockets.connect(upstream_url) as upstream_ws:
            # Create tasks for bidirectional communication
            async def send_to_upstream():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await upstream_ws.send(data)
                except:
                    pass  # Connection closed
            
            async def receive_from_upstream():
                try:
                    while True:
                        data = await upstream_ws.recv()
                        await websocket.send_text(data)
                except:
                    pass  # Connection closed
            
            # Run both tasks concurrently
            await asyncio.gather(send_to_upstream(), receive_from_upstream())
    except Exception as e:
        logger.error(f"Disease dashboard WebSocket error: {e}")
        await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn
    print(f"API Gateway running on http://{settings.GATEWAY_HOST}:{settings.GATEWAY_PORT}")
    print("Docs: http://127.0.0.1:8080/docs")
    uvicorn.run(
        "main:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=settings.DEBUG,
    )

