"""
Pydantic request/response schemas — matches the AgroMobile Flutter app contract.
"""

from typing import Optional
from pydantic import BaseModel, EmailStr


# ── Auth Requests ──

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None
    farm_name: Optional[str] = None
    state_code: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


# ── Auth Responses ──

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    farm_name: Optional[str] = None
    state_code: Optional[str] = None


class AuthResponse(BaseModel):
    """Returned on login / register — tokens + user profile."""
    access_token: str
    refresh_token: str
    user: UserResponse


class TokenResponse(BaseModel):
    """Returned on token refresh."""
    access_token: str
    refresh_token: str


# ── Disease Diagnosis ──

class DiagnosisResponse(BaseModel):
    id: str
    user_id: str
    image_filename: Optional[str] = None
    on_device_label: Optional[str] = None
    on_device_confidence: Optional[float] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    crop: Optional[str] = None
    disease_type: Optional[str] = None
    symptoms: Optional[str] = None
    treatment_json: Optional[str] = None
    uploaded: bool = False
    created_at: Optional[str] = None

