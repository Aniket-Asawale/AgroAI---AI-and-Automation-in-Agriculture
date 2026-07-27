"""
API Gateway routes for Auth Service (port 8002).
Proxies registration, login, refresh, forgot-password, profile, and diagnosis endpoints.
"""

from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File, Body
from pydantic import BaseModel, EmailStr

from config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])

BASE = settings.AUTH_URL
TIMEOUT = settings.REQUEST_TIMEOUT


async def _get_client():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        yield client

# ─── Schemas for Swagger UI ───
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    farm_name: Optional[str] = None
    state_code: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Aniket Asawale",
                "email": "aniket@agroaiapp.me",
                "password": "SecurePassword123!",
                "phone": "+919876543210",
                "farm_name": "Green Acres",
                "state_code": "MH"
            }
        }
    }

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "aniket@agroaiapp.me",
                "password": "SecurePassword123!"
            }
        }
    }

class RefreshRequest(BaseModel):
    refresh_token: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    }

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "aniket@agroaiapp.me"
            }
        }
    }


async def _proxy_raw_body(method: str, path: str, request: Request, client: httpx.AsyncClient):
    """Generic raw body proxy helper."""
    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        r = await client.request(method, f"{BASE}{path}", content=body, headers=headers)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if "application/json" in r.headers.get("content-type", "") else r.text
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Auth service unreachable: {exc}")


async def _proxy_json_payload(method: str, path: str, client: httpx.AsyncClient, payload: BaseModel):
    """Proxy helper that sends the validated Pydantic model as JSON."""
    try:
        body = payload.model_dump_json(exclude_none=True).encode("utf-8")
        r = await client.request(
            method=method,
            url=f"{BASE}{path}",
            content=body,
            headers={"Content-Type": "application/json"}
        )
        if r.status_code not in (200, 201):
            detail = r.text
            if "application/json" in r.headers.get("content-type", ""):
                try: detail = r.json().get("detail", r.text)
                except Exception: pass
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Auth Service unreachable: {exc}")


# ── Auth endpoints ──

@router.post("/register")
async def register(payload: RegisterRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_json_payload("POST", "/auth/register", client, payload)


@router.post("/login")
async def login(payload: LoginRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_json_payload("POST", "/auth/login", client, payload)


@router.post("/refresh")
async def refresh(payload: RefreshRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_json_payload("POST", "/auth/refresh", client, payload)


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest = Body(...), client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_json_payload("POST", "/auth/forgot-password", client, payload)


@router.get("/me")
async def me(request: Request, client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_raw_body("GET", "/auth/me", request, client)


# ── Disease diagnosis storage ──

@router.post("/diagnoses")
async def save_diagnosis(request: Request, client: httpx.AsyncClient = Depends(_get_client)):
    """Proxy multipart diagnosis upload to auth service."""
    auth_header = request.headers.get("authorization")
    try:
        form = await request.form()
        files = {}
        data = {}
        for key, value in form.items():
            if hasattr(value, "read"):  # UploadFile
                content = await value.read()
                files[key] = (value.filename, content, value.content_type or "image/jpeg")
            else:
                data[key] = value

        headers = {"Authorization": auth_header} if auth_header else {}
        r = await client.post(
            f"{BASE}/auth/diagnoses",
            files=files if files else None,
            data=data,
            headers=headers,
        )
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.json().get("detail", r.text))
        return r.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Auth service unreachable: {exc}")


@router.get("/diagnoses")
async def list_diagnoses(request: Request, client: httpx.AsyncClient = Depends(_get_client)):
    return await _proxy_raw_body("GET", "/auth/diagnoses", request, client)

