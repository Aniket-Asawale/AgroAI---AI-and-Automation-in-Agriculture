"""
Plant Disease Detection — FastAPI Service

Endpoints:
    POST /detect            -- Upload an image, get disease prediction + treatment info
    GET  /health            -- Health check + model info
    GET  /crops             -- List available crop-specific models
    GET  /diseases          -- List all known diseases in the knowledge base
    GET  /dataset/stats     -- Dataset capture statistics
    GET  /dataset/export    -- Download dataset archive

Usage:
    cd Plant_Disease_Detection
    uvicorn api:app --reload --port 8003
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch import nn
from torchvision import models, transforms

from user_captures import save_capture, get_stats, export_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent / "plant-disease"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
KB_PATH = SRC_DIR / "disease_kb.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ARCH_INPUT_SIZE = {
    "efficientnet_b4": 380,
    "efficientnet_b0": 224,
    "resnet18": 224,
}
ARCH_PRIORITY = ["efficientnet_b4", "efficientnet_b0", "resnet18"]

# File candidates by crop and architecture.
# Extra checkpoints are auto-discovered from the models folder as well.
MODEL_CANDIDATES = {
    "all": {
        "efficientnet_b4": [
            "efficientnet_b4_all_final.pth",
            "efficientnet_b4_tuned_final.pth",
            "efficientnet_b4_final.pth",
        ],
        "efficientnet_b0": ["efficientnet_b0_final.pth"],
        "resnet18": ["resnet18_final.pth"],
    },
    "corn": {"efficientnet_b4": ["efficientnet_b4_corn.pth"]},
    "rice": {"efficientnet_b4": ["efficientnet_b4_rice.pth"]},
    "wheat": {"efficientnet_b4": ["efficientnet_b4_wheat_tuned.pth", "efficientnet_b4_wheat.pth"]},
    "millet": {"efficientnet_b4": ["efficientnet_b4_millet.pth"]},
    "sugarcane": {"efficientnet_b4": ["efficientnet_b4_sugarcane.pth"]},
}

# ─── Globals loaded at startup ───
disease_kb: dict[str, dict[str, Any]] = {}
# models_registry[crop][arch] -> {"model": model, "idx_to_class": ..., "path": str, "input_size": int}
models_registry: dict[str, dict[str, dict[str, Any]]] = {}

# ─── Schemas ───

class DetectionResult(BaseModel):
    predicted_class: str
    confidence: float
    predicted_crop: Optional[str] = None
    crop: Optional[str] = None
    selected_crop: str
    selected_model_variant: str
    model_used: str
    reliable: bool = True
    warning: Optional[str] = None
    top_predictions: list[dict]
    disease_type: Optional[str] = None
    symptoms: Optional[list[str]] = None
    chemical_treatment: Optional[dict] = None
    organic_treatment: Optional[dict] = None
    precautions: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: list[str]
    architectures_loaded: dict[str, list[str]]
    diseases_in_kb: int


# ─── App ───
app = FastAPI(
    title="Plant Disease Detection API",
    description="EfficientNet-B4 powered plant disease detection with treatment recommendations.",
    version="1.0.0",
)


def _build_model(arch: str, num_classes: int):
    if arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def _infer_arch_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "efficientnet_b4" in name:
        return "efficientnet_b4"
    if "efficientnet_b0" in name:
        return "efficientnet_b0"
    if "resnet18" in name:
        return "resnet18"
    return None


def _candidate_paths() -> dict[str, dict[str, list[Path]]]:
    candidates: dict[str, dict[str, list[Path]]] = {}
    for crop, arch_map in MODEL_CANDIDATES.items():
        candidates[crop] = {}
        for arch, files in arch_map.items():
            candidates[crop][arch] = [MODELS_DIR / f for f in files]

    # Auto-discover additional checkpoints by filename convention.
    for p in MODELS_DIR.glob("*.pth"):
        arch = _infer_arch_from_name(p)
        if not arch:
            continue
        name = p.name.lower()
        crop = "all"
        for known in ["corn", "rice", "wheat", "millet", "sugarcane"]:
            if known in name:
                crop = known
                break
        candidates.setdefault(crop, {}).setdefault(arch, [])
        if p not in candidates[crop][arch]:
            candidates[crop][arch].append(p)
    return candidates


def _load_single_model(path: Path, arch: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if "class_to_idx" not in ckpt:
        raise ValueError(f"Missing class_to_idx in checkpoint {path.name}")
    if "model_state" not in ckpt:
        raise ValueError(f"Missing model_state in checkpoint {path.name}")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = _build_model(arch, len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, idx_to_class


def _transform_for_size(size: int):
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _predict_probs(model, x):
    # Small TTA: original + horizontal flip.
    with torch.no_grad():
        p1 = torch.softmax(model(x), dim=1)
        p2 = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
        probs = (p1 + p2) / 2.0
    return probs[0]


def _infer_crop_from_class(pred_class: str) -> Optional[str]:
    kb_entry = disease_kb.get(pred_class)
    if kb_entry and kb_entry.get("crop"):
        return str(kb_entry["crop"]).strip().lower()
    if "_" in pred_class:
        return pred_class.split("_", 1)[0].strip().lower()
    return None


def _select_arch(crop: str, requested_variant: str) -> str:
    available = models_registry.get(crop, {})
    if not available:
        raise HTTPException(status_code=503, detail=f"No models available for crop '{crop}'")
    if requested_variant != "auto":
        if requested_variant not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Model variant '{requested_variant}' unavailable for crop '{crop}'. Available: {list(available.keys())}",
            )
        return requested_variant
    for arch in ARCH_PRIORITY:
        if arch in available:
            return arch
    return sorted(available.keys())[0]


@app.on_event("startup")
def startup():
    global disease_kb, models_registry
    # Load knowledge base
    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            disease_kb = json.load(f)
        logger.info("Loaded disease KB with %d entries", len(disease_kb))
    else:
        logger.warning("Disease KB not found at %s", KB_PATH)

    # Load models by crop + architecture.
    models_registry = {}
    for crop, arch_map in _candidate_paths().items():
        for arch, paths in arch_map.items():
            loaded = False
            for path in paths:
                if not path.exists():
                    continue
                try:
                    model, idx2cls = _load_single_model(path, arch)
                    models_registry.setdefault(crop, {})[arch] = {
                        "model": model,
                        "idx_to_class": idx2cls,
                        "path": str(path),
                        "input_size": ARCH_INPUT_SIZE.get(arch, 224),
                    }
                    logger.info("Loaded model: crop=%s arch=%s classes=%d file=%s", crop, arch, len(idx2cls), path.name)
                    loaded = True
                    break
                except Exception as exc:
                    logger.warning("Failed loading model %s (%s): %s", path.name, arch, exc)
            if not loaded:
                logger.warning("No usable checkpoint for crop=%s arch=%s", crop, arch)


@app.get("/health", response_model=HealthResponse)
def health():
    arch_summary = {crop: sorted(list(arch_map.keys())) for crop, arch_map in models_registry.items()}
    return HealthResponse(
        status="healthy" if models_registry else "no_models",
        device=str(DEVICE),
        models_loaded=sorted(list(models_registry.keys())),
        architectures_loaded=arch_summary,
        diseases_in_kb=len(disease_kb),
    )


@app.get("/crops")
def list_crops():
    """List available crop-specific models."""
    return {"crops": sorted(list(models_registry.keys())), "total": len(models_registry)}


@app.get("/models")
def list_models():
    out = {}
    for crop, arch_map in models_registry.items():
        out[crop] = [
            {"variant": arch, "file": meta["path"], "input_size": meta["input_size"]}
            for arch, meta in sorted(arch_map.items())
        ]
    return {"models": out}


@app.get("/diseases")
def list_diseases():
    """List all diseases in the knowledge base."""
    entries = []
    for key, info in disease_kb.items():
        entries.append({"key": key, "crop": info.get("crop"), "type": info.get("type")})
    return {"diseases": entries, "total": len(entries)}


@app.post("/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG)"),
    crop: str = Query("all", description="Crop model to use: all, corn, rice, wheat, millet, sugarcane"),
    model_variant: str = Query("auto", description="Model variant: auto, efficientnet_b4, efficientnet_b0, resnet18"),
    min_confidence: float = Query(35.0, ge=0.0, le=100.0, description="Minimum confidence to treat prediction as reliable"),
    save_to_dataset: bool = Query(False, description="Save image to user-contributed dataset"),
):
    """Upload a leaf image and get disease prediction with treatment info."""
    crop = crop.strip().lower()
    model_variant = model_variant.strip().lower()
    if crop not in models_registry:
        raise HTTPException(status_code=400, detail=f"Unknown crop model '{crop}'. Available: {list(models_registry.keys())}")

    # Read & preprocess image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    arch = _select_arch(crop, model_variant)
    model_meta = models_registry[crop][arch]
    model = model_meta["model"]
    idx2cls = model_meta["idx_to_class"]
    transform = _transform_for_size(model_meta["input_size"])

    x = transform(img).unsqueeze(0).to(DEVICE)
    probs = _predict_probs(model, x)
    conf, pred_idx = torch.max(probs, dim=0)
    top_k = min(3, probs.shape[0])
    top_conf, top_idx = torch.topk(probs, k=top_k)

    pred_class = idx2cls[pred_idx.item()]
    confidence = round(conf.item() * 100, 2)
    top_predictions = [
        {"class": idx2cls[i.item()], "confidence": round(c.item() * 100, 2)}
        for c, i in zip(top_conf, top_idx)
    ]

    warning_parts = []
    predicted_crop = _infer_crop_from_class(pred_class)
    if confidence < min_confidence:
        warning_parts.append(f"Low confidence ({confidence}%).")
    if crop != "all" and predicted_crop and predicted_crop != crop:
        warning_parts.append(f"Crop mismatch: selected '{crop}' but predicted class appears to be '{predicted_crop}'.")
    warning = " ".join(warning_parts) if warning_parts else None
    reliable = warning is None

    # Save to user-contributed dataset if requested
    if save_to_dataset:
        try:
            save_capture(
                image_bytes=contents,
                prediction=pred_class,
                confidence=confidence / 100.0,
                original_filename=file.filename or "upload.jpg",
            )
            logger.info("Saved capture to dataset: %s (%.1f%%)", pred_class, confidence)
        except Exception as e:
            logger.warning("Failed to save capture: %s", e)

    # Look up knowledge base
    kb_entry = disease_kb.get(pred_class, {})

    return DetectionResult(
        predicted_class=pred_class,
        confidence=confidence,
        predicted_crop=predicted_crop,
        crop=kb_entry.get("crop"),
        selected_crop=crop,
        selected_model_variant=model_variant,
        model_used=arch,
        reliable=reliable,
        warning=warning,
        top_predictions=top_predictions,
        disease_type=kb_entry.get("type"),
        symptoms=kb_entry.get("symptoms"),
        chemical_treatment=kb_entry.get("chemical_treatment"),
        organic_treatment=kb_entry.get("organic_treatment"),
        precautions=kb_entry.get("precautions"),
    )


@app.get("/dataset/stats")
def dataset_stats():
    """Get statistics about the user-contributed image dataset."""
    return get_stats()


@app.get("/dataset/export")
def dataset_export():
    """Export the user-contributed dataset as a tar.gz archive."""
    stats = get_stats()
    if stats["total_captures"] == 0:
        raise HTTPException(status_code=404, detail="No captures in dataset yet.")
    
    archive_path = export_dataset()
    return FileResponse(
        path=archive_path,
        media_type="application/gzip",
        filename=Path(archive_path).name,
    )

from fastapi.responses import RedirectResponse

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/docs")

