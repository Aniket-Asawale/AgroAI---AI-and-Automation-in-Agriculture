"""LLM-based verifier for crop recommendations.

Wraps Groq (primary, OpenAI-compatible) and Gemini (fallback) chat APIs
to second-opinion the ML model's top-3 output.  All network failures and
parse errors are swallowed and returned as ``error`` so the API never
blocks on a flaky third-party.

Public entry point:
    verify_recommendation(inputs, top_3, ood_info, timeout=8.0) -> dict

Returned dict shape:
    {
        "enabled":   bool,
        "provider":  "groq" | "gemini" | None,
        "model":     str | None,
        "verdict":   "AGREE" | "PARTIAL" | "DISAGREE" | "UNCERTAIN" | None,
        "adjusted_top_3": list | None,    # same shape as input top_3
        "reasoning": str | None,
        "concerns":  list[str],
        "latency_ms": int | None,
        "error":     str | None,
    }
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

_LLM_TIMEOUT_S = 8.0
_MAX_CONFIDENCE = 0.97
_PROMPT_SYS = (
    "You are an experienced Indian agronomist verifying a machine-learning "
    "crop recommendation for a Maharashtra farm. You will receive structured "
    "soil/weather/location inputs and the model's top-3 candidate crops with "
    "calibrated confidences. Reply ONLY with a single JSON object matching "
    "exactly this schema, no prose around it:\n"
    "{\n"
    '  "verdict": "AGREE" | "PARTIAL" | "DISAGREE" | "UNCERTAIN",\n'
    '  "adjusted_top_3": [ {"crop": str, "confidence": float (0..0.97)} ],\n'
    '  "reasoning": str (<=400 chars, plain text),\n'
    '  "concerns": [ str ]   (0..4 short bullet strings)\n'
    "}\n"
    "Rules: never emit confidence > 0.97; if model is uniform/uncertain set "
    "verdict=UNCERTAIN; preserve the same crop names; if you disagree, you "
    "may reorder the same 3 crops or replace one with a better fit for the "
    "given soil/season/zone but keep length=3."
)


def _load_api_key(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val.strip()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for candidate in (".env", ".mobile_env"):
            p = parent / candidate
            if p.is_file():
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        if k.strip() == name and v.strip():
                            return v.strip().strip('"').strip("'")
                except Exception:
                    pass
        for sub in ("AgroModules/AgroMobile", "AgroMobile"):
            p = parent / sub / ".mobile_env"
            if p.is_file():
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        if k.strip() == name and v.strip():
                            return v.strip().strip('"').strip("'")
                except Exception:
                    pass
    return None


def _build_user_payload(inputs: dict, top_3: list, ood_info: dict) -> str:
    keep_in = {k: inputs.get(k) for k in (
        "nitrogen", "phosphorus", "potassium", "ph", "ec",
        "organic_carbon", "moisture", "temperature",
        "weather_temp", "humidity", "rainfall",
        "lat", "lon", "altitude",
        "soil_type", "soil_texture", "drainage",
        "agro_zone", "season", "month",
    ) if k in inputs}
    keep_top = [
        {"crop": c.get("crop"),
         "confidence": round(float(c.get("confidence", 0.0)), 4),
         "raw_confidence": round(float(c.get("raw_confidence", 0.0)), 4)}
        for c in (top_3 or [])
    ]
    keep_ood = {
        "geo_outside": bool((ood_info or {}).get("geo_outside", False)),
        "mahal_distance": (ood_info or {}).get("mahal_distance"),
        "mahal_threshold": (ood_info or {}).get("mahal_threshold"),
    }
    return json.dumps({
        "inputs": keep_in,
        "model_top_3": keep_top,
        "ood": keep_ood,
    }, ensure_ascii=False, default=str)



def _parse_json_strict(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction from a chat completion."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _normalize_verdict(parsed: dict, original_top_3: list) -> dict:
    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in ("AGREE", "PARTIAL", "DISAGREE", "UNCERTAIN"):
        verdict = "UNCERTAIN"
    raw_adj = parsed.get("adjusted_top_3") or []
    adjusted = []
    for entry in raw_adj[:3]:
        try:
            crop = str(entry.get("crop", "")).strip()
            conf = float(entry.get("confidence", 0.0))
        except Exception:
            continue
        if not crop:
            continue
        conf = max(0.0, min(_MAX_CONFIDENCE, conf))
        adjusted.append({"crop": crop, "confidence": round(conf, 4),
                         "confidence_pct": f"{conf * 100:.1f}%"})
    if not adjusted and original_top_3:
        adjusted = [
            {"crop": c.get("crop"),
             "confidence": round(float(c.get("confidence", 0.0)), 4),
             "confidence_pct": c.get("confidence_pct", "0.0%")}
            for c in original_top_3[:3]
        ]
    reasoning = str(parsed.get("reasoning", "")).strip()[:600]
    concerns_raw = parsed.get("concerns") or []
    concerns = [str(c).strip()[:200] for c in concerns_raw if str(c).strip()][:4]
    return {"verdict": verdict, "adjusted_top_3": adjusted,
            "reasoning": reasoning, "concerns": concerns}


def _call_groq(payload_json: str, timeout: float) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (text, model, error)."""
    key = _load_api_key("GROQ_API_KEY")
    if not key:
        return None, None, "GROQ_API_KEY not configured"
    base = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "max_tokens": 600,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": _PROMPT_SYS},
                               {"role": "user", "content": payload_json}]},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None, model, f"groq http {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text, model, None
    except Exception as exc:
        return None, model, f"groq exception: {exc!s}"


def _call_gemini(payload_json: str, timeout: float) -> tuple[Optional[str], Optional[str], Optional[str]]:
    key = _load_api_key("GEMINI_API_KEY") or _load_api_key("GOOGLE_API_KEY")
    if not key:
        return None, None, "GEMINI_API_KEY not configured"
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    try:
        resp = requests.post(
            url, timeout=timeout,
            json={"systemInstruction": {"parts": [{"text": _PROMPT_SYS}]},
                  "contents": [{"role": "user", "parts": [{"text": payload_json}]}],
                  "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600,
                                       "responseMimeType": "application/json"}},
        )
        if resp.status_code != 200:
            return None, model, f"gemini http {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text, model, None
    except Exception as exc:
        return None, model, f"gemini exception: {exc!s}"


def verify_recommendation(inputs: dict, top_3: list, ood_info: Optional[dict] = None,
                          timeout: float = _LLM_TIMEOUT_S) -> dict:
    """Verify ML output against an LLM second opinion. Always returns a dict."""
    result: dict = {"enabled": False, "provider": None, "model": None,
                    "verdict": None, "adjusted_top_3": None,
                    "reasoning": None, "concerns": [],
                    "latency_ms": None, "error": None}
    payload = _build_user_payload(inputs, top_3, ood_info or {})
    t0 = time.perf_counter()
    text, model, err = _call_groq(payload, timeout)
    provider = "groq"
    if text is None:
        text2, model2, err2 = _call_gemini(payload, timeout)
        if text2 is not None:
            text, model, err, provider = text2, model2, None, "gemini"
        else:
            result["error"] = f"{err}; {err2}"
            return result
    result["enabled"] = True
    result["provider"] = provider
    result["model"] = model
    parsed = _parse_json_strict(text or "")
    if not parsed:
        result["error"] = "could not parse JSON from LLM"
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return result
    norm = _normalize_verdict(parsed, top_3)
    result.update(norm)
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return result
