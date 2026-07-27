"""
SQLAlchemy ORM models — Users & Disease Diagnosis History.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Boolean, DateTime, Float, Text, ForeignKey, LargeBinary,
)
from sqlalchemy.orm import relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)
    farm_name = Column(String(200), nullable=True)
    state_code = Column(String(10), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    diagnoses = relationship("DiagnosisRecord", back_populates="user", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "farm_name": self.farm_name,
            "state_code": self.state_code,
        }


class DiagnosisRecord(Base):
    """Stores plant disease diagnosis results with optional image data."""
    __tablename__ = "disease_diagnoses"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    image_data = Column(LargeBinary, nullable=True)  # stored image bytes
    image_filename = Column(String(255), nullable=True)
    image_content_type = Column(String(50), nullable=True)

    # On-device inference result
    on_device_label = Column(String(200), nullable=True)
    on_device_confidence = Column(Float, nullable=True)

    # Server inference result
    predicted_class = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    crop = Column(String(100), nullable=True)
    disease_type = Column(String(100), nullable=True)
    symptoms = Column(Text, nullable=True)           # JSON string
    treatment_json = Column(Text, nullable=True)      # JSON string

    uploaded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="diagnoses")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_filename": self.image_filename,
            "on_device_label": self.on_device_label,
            "on_device_confidence": self.on_device_confidence,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "crop": self.crop,
            "disease_type": self.disease_type,
            "symptoms": self.symptoms,
            "treatment_json": self.treatment_json,
            "uploaded": self.uploaded,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

