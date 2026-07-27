"""
LLM-based Crop Label Validator for the Crop Recommendation Engine.
Validates 'uncertain' confidence rows using Groq (primary), Gemini, or Mistral.

Usage:
  1. Set API key(s): GROQ_API_KEY, GEMINI_API_KEY, and/or MISTRAL_API_KEY
  2. python Crop_Recommendation_Engine/generators/crop_label_validator.py

Reads:  data/datasets/crop_recommendation_dataset.csv
Writes: data/datasets/crop_recommendation_dataset.csv (updated in-place)
        data/processed/llm_validation_report.txt
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Load env files (check .global_env at repo root, then AgroSensor/.env)
_ROOT = Path(__file__).resolve().parent.parent.parent
for _env_file in [_ROOT / ".global_env", _ROOT / "AgroSensor" / ".env"]:
    if _env_file.exists():
        with open(_env_file, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _val = _val.strip()
                    if _val and not os.getenv(_key.strip()):
                        os.environ[_key.strip()] = _val

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CSV_PATH = DATA_DIR / "crop_recommendation_dataset.csv"
REPORT_PATH = PROCESSED_DIR / "llm_validation_report.txt"

BATCH_SIZE = 25  # rows per LLM call (keep prompt within free-tier token limits)
RATE_LIMIT_DELAY = 4  # seconds between API calls (Groq: ~17 RPM, safe at 4s)
MAX_RETRIES = 3  # max retries per batch on 429/rate-limit errors
INITIAL_BACKOFF = 30  # initial backoff seconds on 429 error

# ─── Prompt Template ───
SYSTEM_PROMPT = """You are an Indian agriculture expert specializing in Maharashtra and the Deccan Plateau.
You validate crop recommendation data. For each row, check:
1. Is the crop label correct for this soil type + season + location (city/agro_zone)?
2. Are NPK values realistic for this soil type in this region?
3. Would this crop actually grow in this location during this season?

Respond ONLY with valid JSON — an array of objects:
[{"row_id": <int>, "valid": true/false, "suggested_crop": "<if invalid, suggest correct crop or null>", "reason": "<brief reason>"}]

Do NOT include any text outside the JSON array."""


def _build_batch_prompt(rows: list[dict]) -> str:
    """Build a validation prompt for a batch of rows."""
    header = "Validate these crop recommendation rows:\n\n"
    header += "row_id | city | state | agro_zone | soil_type | season | "
    header += "N | P | K | pH | crop_label\n"
    header += "-" * 100 + "\n"

    lines = []
    for r in rows:
        lines.append(
            f"{r['_row_id']} | {r['city']} | {r['state']} | {r['agro_zone']} | "
            f"{r['soil_type']} | {r['season']} | "
            f"{r['sensor_nitrogen']} | {r['sensor_phosphorus']} | {r['sensor_potassium']} | "
            f"{r['sensor_ph']} | {r['crop_label']}"
        )

    return header + "\n".join(lines)


# ─── LLM Backends ───

def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Google Gemini API (2.5 Flash-Lite — free tier, not deprecated)."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=SYSTEM_PROMPT + "\n\n" + prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 4096,
        },
    )
    return response.text


def _call_groq(prompt: str, api_key: str) -> str:
    """Call Groq API (Llama 3.3 70B — 1,000 RPD free tier)."""
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _call_mistral(prompt: str, api_key: str) -> str:
    """Call Mistral API."""
    from mistralai import Mistral

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_llm_response(text: str) -> list[dict]:
    """Parse JSON array from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from the text
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Debug: show full response length and boundaries
    print(f"  WARNING: Could not parse LLM response (len={len(text)}):")
    print(f"  FIRST 300 chars: {repr(text[:300])}")
    print(f"  LAST 300 chars: {repr(text[-300:])}")
    return []


def _select_backend():
    """Select available LLM backend and return (call_fn, api_key, name). Prioritizes Groq."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    mistral_key = os.getenv("MISTRAL_API_KEY", "")

    if groq_key:
        return _call_groq, groq_key, "Groq Llama 3.3 70B"
    elif gemini_key:
        return _call_gemini, gemini_key, "Gemini 2.5 Flash"
    elif mistral_key:
        return _call_mistral, mistral_key, "Mistral"
    else:
        print("ERROR: No API key found.")
        print("Set GROQ_API_KEY, GEMINI_API_KEY, or MISTRAL_API_KEY environment variable.")
        sys.exit(1)


def validate_uncertain_rows():
    """Main validation pipeline: extract uncertain rows, send to LLM, update dataset."""
    call_fn, api_key, backend_name = _select_backend()

    # Load dataset
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames

    # Find uncertain rows
    uncertain_indices = []
    for i, row in enumerate(all_rows):
        if row.get("confidence_label") == "uncertain":
            uncertain_indices.append(i)

    print(f"Found {len(uncertain_indices)} uncertain rows out of {len(all_rows)} total")
    if not uncertain_indices:
        print("No uncertain rows to validate.")
        return

    # Prepare batches
    batches = []
    for i in range(0, len(uncertain_indices), BATCH_SIZE):
        batch_indices = uncertain_indices[i : i + BATCH_SIZE]
        batch_rows = []
        for idx in batch_indices:
            row = all_rows[idx].copy()
            row["_row_id"] = idx
            batch_rows.append(row)
        batches.append((batch_indices, batch_rows))

    print(f"Processing {len(batches)} batches of up to {BATCH_SIZE} rows each")
    print(f"Using LLM backend: {backend_name}")
    print()

    # Process each batch
    total_valid = 0
    total_invalid = 0
    total_errors = 0
    corrections = []
    report_lines = [
        "=" * 60,
        f"LLM VALIDATION REPORT — {backend_name}",
        "=" * 60,
        f"Total uncertain rows: {len(uncertain_indices)}",
        f"Batch size: {BATCH_SIZE}",
        "",
    ]

    for batch_num, (batch_indices, batch_rows) in enumerate(batches, 1):
        print(f"  Batch {batch_num}/{len(batches)} ({len(batch_rows)} rows)...", end=" ")

        prompt = _build_batch_prompt(batch_rows)
        try:
            # Retry loop with exponential backoff for rate limit (429) errors
            response_text = None
            for attempt in range(MAX_RETRIES):
                try:
                    response_text = call_fn(prompt, api_key)
                    break  # success
                except Exception as retry_err:
                    err_str = str(retry_err)
                    is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        backoff = INITIAL_BACKOFF * (2 ** attempt)
                        print(f"\n    ⏳ Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {backoff}s...", end=" ")
                        time.sleep(backoff)
                    else:
                        raise  # re-raise non-429 errors or final attempt

            if response_text is None:
                raise RuntimeError("All retry attempts exhausted")

            results = _parse_llm_response(response_text)

            batch_valid = 0
            batch_invalid = 0

            for result in results:
                row_id = result.get("row_id")
                valid = result.get("valid", True)
                suggested = result.get("suggested_crop")
                reason = result.get("reason", "")

                if row_id is None or row_id not in batch_indices:
                    continue

                if valid:
                    batch_valid += 1
                    # Upgrade confidence from uncertain to medium (LLM-confirmed)
                    all_rows[row_id]["confidence_label"] = "medium"
                    flag = all_rows[row_id].get("data_quality_flag", "")
                    all_rows[row_id]["data_quality_flag"] = f"{flag}|llm_validated" if flag else "llm_validated"
                else:
                    batch_invalid += 1
                    old_crop = all_rows[row_id]["crop_label"]
                    if suggested and suggested != old_crop:
                        all_rows[row_id]["crop_label"] = suggested
                        all_rows[row_id]["confidence_label"] = "medium"
                        all_rows[row_id]["data_quality_flag"] = f"llm_corrected:{old_crop}->{suggested}"
                        corrections.append(
                            f"  Row {row_id}: {old_crop} → {suggested} ({reason})"
                        )
                    else:
                        # LLM says invalid but no suggestion — keep as uncertain
                        all_rows[row_id]["data_quality_flag"] = f"llm_flagged:{reason[:50]}"

            total_valid += batch_valid
            total_invalid += batch_invalid
            print(f"✓ {batch_valid} valid, ✗ {batch_invalid} invalid")

            report_lines.append(
                f"Batch {batch_num}: {batch_valid} valid, {batch_invalid} invalid"
            )

        except Exception as e:
            total_errors += 1
            print(f"ERROR: {e}")
            report_lines.append(f"Batch {batch_num}: ERROR — {e}")

        # Rate limiting
        if batch_num < len(batches):
            time.sleep(RATE_LIMIT_DELAY)

    # Save updated dataset
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    report_lines.extend([
        "",
        "── Summary ──",
        f"  LLM-validated (valid):   {total_valid}",
        f"  LLM-flagged (invalid):   {total_invalid}",
        f"  API errors:              {total_errors}",
        f"  Crop corrections:        {len(corrections)}",
    ])
    if corrections:
        report_lines.append("\n── Corrections ──")
        report_lines.extend(corrections)

    # Recount confidence distribution
    conf_counts = {}
    for row in all_rows:
        cl = row.get("confidence_label", "unknown")
        conf_counts[cl] = conf_counts.get(cl, 0) + 1
    report_lines.append("\n── Updated Confidence Distribution ──")
    for label, count in sorted(conf_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {label}: {count} ({count/len(all_rows)*100:.1f}%)")

    report_text = "\n".join(report_lines)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n✅ Updated dataset saved to {CSV_PATH}")
    print(f"✅ Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    validate_uncertain_rows()
