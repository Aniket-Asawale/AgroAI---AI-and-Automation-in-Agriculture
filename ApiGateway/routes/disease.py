"""
API Gateway routes for Plant Disease Detection service (port 8003).
Proxies detection, health, and listing endpoints.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Request
from fastapi.responses import StreamingResponse

from config import settings

router = APIRouter(prefix="/disease", tags=["Plant Disease Detection"])

BASE = settings.PLANT_DISEASE_URL
TIMEOUT = settings.REQUEST_TIMEOUT


async def _get_client():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client


@router.get("/health")
async def disease_health(client: httpx.AsyncClient = Depends(_get_client)):
    """Plant Disease Detection health check."""
    try:
        r = await client.get(f"{BASE}/health")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Plant Disease service unreachable: {exc}")


@router.get("/crops")
async def disease_crops(client: httpx.AsyncClient = Depends(_get_client)):
    """List available crop models for disease detection."""
    try:
        r = await client.get(f"{BASE}/crops")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Plant Disease service unreachable: {exc}")


@router.get("/diseases")
async def list_diseases(client: httpx.AsyncClient = Depends(_get_client)):
    """List all known diseases in the knowledge base."""
    try:
        r = await client.get(f"{BASE}/diseases")
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Plant Disease service unreachable: {exc}")


@router.post("/detect")
async def detect(
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    crop: str = Query("all", description="Crop model: all, corn, rice, wheat, millet, sugarcane"),
    client: httpx.AsyncClient = Depends(_get_client),
):
    """Upload a leaf image and get disease prediction with treatment info."""
    try:
        contents = await file.read()
        files = {"file": (file.filename, contents, file.content_type or "image/jpeg")}
        r = await client.post(f"{BASE}/detect", files=files, params={"crop": crop})
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Plant Disease service unreachable: {exc}")


@router.get("/dashboard/{path:path}")
@router.post("/dashboard/{path:path}")
async def disease_dashboard_proxy(
    path: str,
    request: Request,
    client: httpx.AsyncClient = Depends(_get_client)
):
    """Proxy requests to disease dashboard (Gradio) on port 7860."""
    try:
        query_string = str(request.url.query) if request.url.query else ""
        target_url = f"http://127.0.0.1:7860/{path}"
        if query_string:
            target_url += f"?{query_string}"
        
        body = None
        if request.method == "POST":
            try:
                body = await request.body()
            except:
                body = None
        
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
        raise HTTPException(status_code=502, detail=f"Disease dashboard unavailable: {exc}")

