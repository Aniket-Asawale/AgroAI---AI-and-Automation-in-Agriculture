"""
Auth Service — FastAPI + PostgreSQL

Endpoints:
    POST /auth/register        — Create account → tokens + user
    POST /auth/login           — Email/password → tokens + user
    POST /auth/refresh         — Refresh token → new token pair
    POST /auth/forgot-password — Placeholder for password reset
    GET  /auth/me              — Get current user profile (Bearer token)
    POST /auth/diagnoses       — Save a disease diagnosis record (with image)
    GET  /auth/diagnoses       — List user's diagnosis history
    GET  /health               — Health check

Usage:
    cd Auth
    uvicorn main:app --reload --port 8002
"""

import json
import logging
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config import settings
from database import get_db, create_tables
from models import User, DiagnosisRecord
from auth_utils import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
)
from schemas import (
    RegisterRequest, LoginRequest, RefreshRequest, ForgotPasswordRequest,
    AuthResponse, TokenResponse, UserResponse, DiagnosisResponse,
)

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgroModules Auth Service",
    description="User authentication and disease diagnosis storage with PostgreSQL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Creating database tables...")
    create_tables()
    logger.info("Auth service ready on port %d", settings.PORT)


# ── Dependency: extract current user from Bearer token ──

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ═══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        name=req.name,
        email=req.email,
        hashed_password=hash_password(req.password),
        phone=req.phone,
        farm_name=req.farm_name,
        state_code=req.state_code,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=UserResponse(**user.to_dict()),
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=UserResponse(**user.to_dict()),
    )


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@app.post("/auth/forgot-password", status_code=200)
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    # Placeholder — in production, send a reset email
    user = db.query(User).filter(User.email == req.email).first()
    # Always return success to avoid email enumeration
    return {"message": "If that email exists, a reset link has been sent."}


@app.get("/auth/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return UserResponse(**user.to_dict())


# ═══════════════════════════════════════════════════════════════
# DISEASE DIAGNOSIS STORAGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/auth/diagnoses", response_model=DiagnosisResponse, status_code=201)
async def save_diagnosis(
    image: Optional[UploadFile] = File(None),
    on_device_label: Optional[str] = Form(None),
    on_device_confidence: Optional[float] = Form(None),
    predicted_class: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    crop: Optional[str] = Form(None),
    disease_type: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
    treatment_json: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a plant disease diagnosis record with optional image."""
    image_data = None
    image_filename = None
    image_content_type = None
    if image:
        image_data = await image.read()
        image_filename = image.filename
        image_content_type = image.content_type

    record = DiagnosisRecord(
        user_id=user.id,
        image_data=image_data,
        image_filename=image_filename,
        image_content_type=image_content_type,
        on_device_label=on_device_label,
        on_device_confidence=on_device_confidence,
        predicted_class=predicted_class,
        confidence=confidence,
        crop=crop,
        disease_type=disease_type,
        symptoms=symptoms,
        treatment_json=treatment_json,
        uploaded=True,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return DiagnosisResponse(**record.to_dict())


@app.get("/auth/diagnoses", response_model=list[DiagnosisResponse])
def list_diagnoses(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all diagnosis records for the current user."""
    records = (
        db.query(DiagnosisRecord)
        .filter(DiagnosisRecord.user_id == user.id)
        .order_by(DiagnosisRecord.created_at.desc())
        .all()
    )
    return [DiagnosisResponse(**r.to_dict()) for r in records]


# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════

from fastapi.responses import RedirectResponse

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "auth", "port": settings.PORT}


if __name__ == "__main__":
    import uvicorn
    print(f"Auth Service running on http://{settings.HOST}:{settings.PORT}")
    print("Docs: http://127.0.0.1:8002/docs")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

