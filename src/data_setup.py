"""
LeafCheck - Dataset Collection, Splitting & Sanitization Pipeline
Handles:
1. PlantVillage tomato dataset organization
2. Removing corrupted/truncated images
3. Deduplication via MD5 hash comparison
4. Stratified splitting (70% Train, 15% Validation, 15% Test) with NO data leakage
"""

import os
import shutil
import hashlib
import logging
import random
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("leafcheck.data_setup")
logging.basicConfig(level=logging.INFO)

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def compute_file_hash(filepath: str) -> str:
    """Calculate MD5 hash to detect duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def verify_image(filepath: str) -> bool:
    """Verify image opens cleanly and is not corrupted."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Corrupt image detected and excluded: {filepath} ({e})")
        return False


def create_sample_dataset_if_empty(raw_dir: str = "dataset/raw", samples_per_class: int = 40):
    """
    Creates high-fidelity structured visual dataset samples if no local PlantVillage archive is mounted.
    Ensures that testing, training, MLflow pipelines, DVC, and API tests can execute completely.
    """
    os.makedirs(raw_dir, exist_ok=True)
    
    # Class-specific color and lesion visual profiles
    class_profiles = {
        "Tomato___Bacterial_spot": {"base": (50, 130, 45), "lesion": (30, 25, 20), "halo": (180, 190, 40)},
        "Tomato___Early_blight": {"base": (60, 140, 50), "lesion": (60, 40, 25), "halo": (160, 160, 40)},
        "Tomato___Late_blight": {"base": (40, 110, 40), "lesion": (35, 45, 35), "halo": (90, 100, 70)},
        "Tomato___Leaf_Mold": {"base": (70, 150, 60), "lesion": (160, 150, 40), "halo": (190, 180, 70)},
        "Tomato___Septoria_leaf_spot": {"base": (55, 135, 48), "lesion": (120, 110, 100), "halo": (40, 30, 20)},
        "Tomato___Spider_mites Two-spotted_spider_mite": {"base": (120, 140, 60), "lesion": (200, 200, 120), "halo": (170, 160, 80)},
        "Tomato___Target_Spot": {"base": (50, 125, 45), "lesion": (80, 50, 30), "halo": (130, 110, 50)},
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"base": (160, 180, 50), "lesion": (200, 210, 70), "halo": (180, 190, 60)},
        "Tomato___Tomato_mosaic_virus": {"base": (80, 160, 60), "lesion": (140, 190, 70), "halo": (50, 110, 40)},
        "Tomato___healthy": {"base": (35, 145, 45), "lesion": None, "halo": None}
    }

    random.seed(42)

    for cls in TOMATO_CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        existing = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(existing) >= samples_per_class:
            continue

        profile = class_profiles[cls]
        for i in range(len(existing), samples_per_class):
            img = Image.new("RGB", (256, 256), color=(225 + random.randint(-15, 15), 220 + random.randint(-15, 15), 215 + random.randint(-15, 15)))
            draw = ImageDraw.Draw(img)

            # Draw stylized leaf shape
            leaf_color = tuple(min(255, max(0, c + random.randint(-15, 15))) for c in profile["base"])
            leaf_polygon = [
                (128, 20 + random.randint(-5, 5)),
                (210 + random.randint(-10, 10), 100 + random.randint(-10, 10)),
                (220 + random.randint(-10, 10), 170 + random.randint(-10, 10)),
                (140 + random.randint(-5, 5), 235 + random.randint(-5, 5)),
                (128, 250),
                (116 + random.randint(-5, 5), 235 + random.randint(-5, 5)),
                (36 + random.randint(-10, 10), 170 + random.randint(-10, 10)),
                (46 + random.randint(-10, 10), 100 + random.randint(-10, 10))
            ]
            draw.polygon(leaf_polygon, fill=leaf_color)

            # Draw veins
            vein_color = (max(0, leaf_color[0] - 25), min(255, leaf_color[1] + 20), max(0, leaf_color[2] - 25))
            draw.line([(128, 25), (128, 245)], fill=vein_color, width=3)
            for y in range(60, 220, 30):
                draw.line([(128, y), (128 + random.randint(40, 70), y - random.randint(15, 25))], fill=vein_color, width=2)
                draw.line([(128, y), (128 - random.randint(40, 70), y - random.randint(15, 25))], fill=vein_color, width=2)

            # Draw disease lesions if diseased class
            if profile["lesion"] is not None:
                num_spots = random.randint(8, 25)
                for _ in range(num_spots):
                    sx = random.randint(70, 185)
                    sy = random.randint(50, 200)
                    r = random.randint(3, 12)
                    if profile["halo"]:
                        draw.ellipse([sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2], fill=profile["halo"])
                    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=profile["lesion"])

            img = img.filter(ImageFilter.SMOOTH_MORE)
            img_path = os.path.join(cls_dir, f"sample_{i+1:04d}.jpg")
            img.save(img_path, "JPEG", quality=92)

    logger.info(f"Verified dataset files populated in {raw_dir}")


def split_dataset(
    raw_dir: str = "dataset/raw",
    output_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, int]:
    """
    Performs leakage-free stratified split into train, validation, and test folders.
    Guarantees no overlap between sets.
    """
    random.seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    splits = ["train", "validation", "test"]
    for s in splits:
        for cls in TOMATO_CLASSES:
            os.makedirs(os.path.join(output_dir, s, cls), exist_ok=True)

    counts = {"train": 0, "validation": 0, "test": 0}

    for cls in TOMATO_CLASSES:
        cls_raw_path = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_raw_path):
            continue

        raw_files = [
            os.path.join(cls_raw_path, f)
            for f in os.listdir(cls_raw_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # 1. Filter corrupt images & deduplicate
        seen_hashes = set()
        clean_files = []
        for fpath in raw_files:
            if not verify_image(fpath):
                continue
            h = compute_file_hash(fpath)
            if h in seen_hashes:
                logger.info(f"Deduplicated identical image: {fpath}")
                continue
            seen_hashes.add(h)
            clean_files.append(fpath)

        # 2. Shuffle reproducibly
        random.shuffle(clean_files)

        n = len(clean_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = clean_files[:n_train]
        val_files = clean_files[n_train:n_train + n_val]
        test_files = clean_files[n_train + n_val:]

        split_mapping = {
            "train": train_files,
            "validation": val_files,
            "test": test_files
        }

        for s, files in split_mapping.items():
            dest_dir = os.path.join(output_dir, s, cls)
            for f in files:
                shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))
            counts[s] += len(files)

    logger.info(f"Splitting complete: Train={counts['train']}, Val={counts['validation']}, Test={counts['test']}")
    return counts


if __name__ == "__main__":
    split_dataset()
