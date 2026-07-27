import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import hashlib

# =============== CONFIG ===============
TARGET_ROOT = Path(r"C:\Users\jayes\plant_disease\data_cls\images")
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

# Target per-class sizes (if enough data)
TEST_PER_CLASS = 100
VAL_PER_CLASS = 70

ROOTS = {
    "sugarcane": Path(r"C:\Users\jayes\Desktop\jayesh\major project\dataset\sugarcane data"),
    "rice":      Path(r"C:\Users\jayes\Desktop\jayesh\major project\dataset\rice data"),
    "corn":      Path(r"C:\Users\jayes\Desktop\jayesh\major project\dataset\corn data"),
    "millet":    Path(r"C:\Users\jayes\Desktop\jayesh\major project\dataset\millet data"),
    "wheat":     Path(r"C:\Users\jayes\Desktop\jayesh\major project\dataset\wheat data"),
}

FOLDER_TO_CLASS = {
    # sugarcane
    "healthy":        "sugarcane_healthy",
    "mosaic":         "sugarcane_mosaic",
    "redrot":         "sugarcane_redrot",
    "rust":           "sugarcane_rust",
    "yellow":         "sugarcane_yellow",

    # rice
    "bacterialblight": "rice_bacterialblight",
    "brownspot":       "rice_brownspot",
    "leafsmut":        "rice_leafsmut",

    # millet
    "blast":          "millet_blast",       
    "rust":           "millet_rust",
    "healthy_millet": "millet_healthy",

    # corn
    "blight":         "corn_blight",
    "common_rust":    "corn_common_rust",
    "gray_leaf_spot": "corn_gray_spot",
    "gray_leaf-spot": "corn_gray_spot",
    "gray_leaf":      "corn_gray_spot",
    "healthy_corn":   "corn_healthy",

    # wheat
    "aphid":                "wheat_aphid",
    "black_rust":           "wheat_black_rust",
    "brown_rust":           "wheat_brown_rust",
    "common_root_rot":      "wheat_common_root_rot",
    "fusarium_head_blight": "wheat_fusarium_head_blight",
    "healthy":              "wheat_healthy",
    "leaf_blight":          "wheat_leaf_blight",
    "mildew":               "wheat_mildew",
    "mite":                 "wheat_mite",
    "septoria":             "wheat_septoria",
    "smut":                 "wheat_smut",
    "stem_fly":             "wheat_stem_fly",
    "tan_spot":             "wheat_tan_spot",
    "yellow_rust":          "wheat_yellow_rust",
    "blast_wheat":          "wheat_blast",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Global hash set: avoid duplicates across entire dataset
GLOBAL_HASH_SET = set()


# =============== HELPERS ===============
def get_image_hash(img_path: Path) -> str:
    """MD5 hash of image bytes for exact duplicate detection."""
    h = hashlib.md5()
    with open(img_path, "rb") as f:
        while chunk := f.read(4096):
            h.update(chunk)
    return h.hexdigest()


def copy_split_images(class_name: str, images: list[Path]):
    """Split images into train/val/test and copy into TARGET_ROOT."""
    random.shuffle(images)

    test_imgs = images[:TEST_PER_CLASS]
    val_imgs = images[TEST_PER_CLASS:TEST_PER_CLASS + VAL_PER_CLASS]
    train_imgs = images[TEST_PER_CLASS + VAL_PER_CLASS:]

    splits = {
        "test": test_imgs,
        "val": val_imgs,
        "train": train_imgs,
    }

    for split_name, img_list in splits.items():
        target_dir = TARGET_ROOT / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            dest = target_dir / img_path.name
            counter = 1
            while dest.exists():
                dest = target_dir / f"{img_path.stem}_dup{counter}{img_path.suffix}"
                counter += 1
            shutil.copy2(img_path, dest)

    print(
        f"[OK] {class_name}: "
        f"{len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test"
    )


def collect_class_images_for_crop(crop: str, root: Path) -> dict[str, list[Path]]:
    """
    Return a dict: class_name -> list[Path] of images (with duplicates removed globally).
    Handles wheat's pre-split folders specially.
    """
    print(f"\n=== Processing {crop} at {root} ===")
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        return {}

    class_images: dict[str, list[Path]] = defaultdict(list)

    if crop == "wheat":
        # Wheat has nested folders like aphid_train, blast_valid, etc.
        for folder in root.rglob("*"):
            if not folder.is_dir():
                continue

            folder_name = folder.name.lower()
            matched_class = None
            for key, mapped_class in FOLDER_TO_CLASS.items():
                if key in folder_name and mapped_class.startswith("wheat_"):
                    matched_class = mapped_class
                    break

            if matched_class:
                images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
                if images:
                    class_images[matched_class].extend(images)

    else:
        # Other crops: only match keys whose mapped class starts with this crop prefix
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            folder_name = folder.name.lower()

            mapped_class = None
            for key, cls in FOLDER_TO_CLASS.items():
                if key == folder_name and cls.startswith(f"{crop}_"):
                    mapped_class = cls
                    break

            if mapped_class:
                images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
                if images:
                    class_images[mapped_class].extend(images)

    # Remove duplicates by hash (within and across crops)
    clean_class_images: dict[str, list[Path]] = {}
    for class_name, imgs in class_images.items():
        print(f"  Raw {class_name}: {len(imgs)} images")
        unique_imgs = []
        seen_hashes_local = set()

        for img in imgs:
            try:
                img_hash = get_image_hash(img)
            except Exception as e:
                print(f"    [WARN] Failed to hash {img}: {e}")
                continue

            if img_hash in GLOBAL_HASH_SET or img_hash in seen_hashes_local:
                continue

            GLOBAL_HASH_SET.add(img_hash)
            seen_hashes_local.add(img_hash)
            unique_imgs.append(img)

        print(f"  Unique {class_name}: {len(unique_imgs)} images after dedup")
        clean_class_images[class_name] = unique_imgs

    return clean_class_images


# =============== MAIN ===============
if __name__ == "__main__":
    random.seed(42)

    # Optional: clear existing dataset (uncomment once if you want a clean rebuild)
    # shutil.rmtree(TARGET_ROOT, ignore_errors=True)
    # TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    all_class_images: dict[str, list[Path]] = defaultdict(list)

    # 1. Collect all class->images from all crops, with dedup
    for crop, root in ROOTS.items():
        crop_class_images = collect_class_images_for_crop(crop, root)
        for cls, imgs in crop_class_images.items():
            all_class_images[cls].extend(imgs)

    # 2. For each class, do stratified split and copy
    print("\n=== STRATIFIED SPLITS ===")
    for class_name, imgs in all_class_images.items():
        total = len(imgs)
        needed = TEST_PER_CLASS + VAL_PER_CLASS
        if total < needed:
            print(f"[WARN] {class_name}: only {total} unique images (need ≥ {needed}), skipping balanced split")
            continue

        copy_split_images(class_name, imgs)

    # 3. Summary
    print("\n=== SUMMARY ===")
    for split in ["train", "val", "test"]:
        split_dir = TARGET_ROOT / split
        if not split_dir.exists():
            print(f"{split}: directory not found")
            continue
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        total_imgs = sum(len(list(d.glob("*"))) for d in class_dirs)
        print(f"{split}: {len(class_dirs)} classes, {total_imgs} images")
