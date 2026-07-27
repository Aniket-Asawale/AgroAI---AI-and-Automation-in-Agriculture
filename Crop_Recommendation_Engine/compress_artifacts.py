"""
Compress model artifacts to match the compressed model variant.
Creates *_compressed.pkl versions of scaler, ood_stats, and conformal files
so the GitHub/Streamlit deployment can ship a single self-consistent set of
*_compressed.pkl artifacts (and inference.py will auto-prefer them).
"""

import joblib
from pathlib import Path
from config import REGISTRY_DIR, MODEL_STAMP

# lzma at level 9 matches what compress_model.py uses for the model + calibrator.
# These artifacts are tiny (scaler is ~2 KB, ood_stats ~25 KB) so the encoder
# choice barely matters, but staying on lzma keeps the deploy bundle uniform.
COMPRESS = ("lzma", 9)


def compress_artifact(filename: str, overwrite: bool = True) -> Path | None:
    """Load an artifact and save a compressed sibling."""
    src_path = REGISTRY_DIR / filename
    if not src_path.exists():
        print(f"  [skip] Source file not found: {src_path.name}")
        return None

    # scaler_2026_05.pkl -> scaler_2026_05_compressed.pkl
    dst_path = REGISTRY_DIR / f"{src_path.stem}_compressed{src_path.suffix}"

    if dst_path.exists() and not overwrite:
        print(f"  [skip] Already exists: {dst_path.name}")
        return dst_path

    data = joblib.load(src_path)
    joblib.dump(data, dst_path, compress=COMPRESS)

    src_kb = src_path.stat().st_size / 1024
    dst_kb = dst_path.stat().st_size / 1024
    reduction = (1 - dst_kb / src_kb) * 100 if src_kb else 0.0
    print(f"  [OK] {src_path.name} ({src_kb:.1f} KB) -> {dst_path.name} ({dst_kb:.1f} KB, {reduction:.1f}% smaller)")

    return dst_path


def main():
    print(f"Compressing artifacts in: {REGISTRY_DIR}\n")

    artifacts = [
        f"scaler_{MODEL_STAMP}.pkl",
        f"ood_stats_{MODEL_STAMP}.pkl",
        f"conformal_{MODEL_STAMP}.pkl",
    ]

    for artifact in artifacts:
        print(f"Processing: {artifact}")
        compress_artifact(artifact)

    print("\nDone! Compressed artifacts created.")


if __name__ == "__main__":
    main()
