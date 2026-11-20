"""Validate and summarize a newly labeled LeafCheck image batch."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("leafcheck.prepare_new_data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_classes(classes_path: Path) -> list[str]:
    with classes_path.open(encoding="utf-8") as classes_file:
        classes = json.load(classes_file)
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(class_name, str) or not class_name for class_name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(f"Expected a non-empty list of unique class names in {classes_path}")
    return classes


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_new_batch(
    batch_dir: str = "dataset/new_batch",
    classes_path: str = "models/classes.json",
) -> dict[str, Any]:
    """Validate a class-folder image batch and return a data-quality summary."""
    root = Path(batch_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Incoming batch directory does not exist: {root}")

    classes = _load_classes(Path(classes_path))
    distribution = {class_name: 0 for class_name in classes}
    errors: list[str] = []
    duplicate_images: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    discovered_image_count = 0
    valid_image_count = 0
    unique_image_count = 0

    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            if entry.is_file():
                if entry.suffix.lower() in IMAGE_EXTENSIONS:
                    discovered_image_count += 1
                    errors.append(
                        f"Image has no class label (expected class/image layout): {entry}"
                    )
                else:
                    errors.append(f"Unsupported file outside a class folder: {entry}")
            continue
        if entry.name not in distribution:
            errors.append(f"Unknown class label: {entry.name}")
            discovered_image_count += sum(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in entry.rglob("*")
                if not path.name.startswith(".")
            )
            continue

        for image_path in sorted(entry.rglob("*")):
            if not image_path.is_file() or image_path.name.startswith("."):
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                errors.append(f"Unsupported file in class folder: {image_path}")
                continue

            discovered_image_count += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
                content_hash = _file_hash(image_path)
            except (OSError, ValueError) as exc:
                errors.append(f"Unreadable or corrupt image {image_path}: {exc}")
                continue

            valid_image_count += 1
            previous_path = seen_hashes.get(content_hash)
            if previous_path is not None:
                duplicate_images.append(
                    {"duplicate": str(image_path), "original": previous_path}
                )
                continue

            seen_hashes[content_hash] = str(image_path)
            unique_image_count += 1
            distribution[entry.name] += 1

    if discovered_image_count == 0:
        errors.append(f"No supported images found in incoming batch: {root}")

    return {
        "batch_dir": str(root),
        "is_valid": not errors,
        "dataset_size": discovered_image_count,
        "valid_image_count": valid_image_count,
        "unique_image_count": unique_image_count,
        "duplicate_count": len(duplicate_images),
        "class_distribution": distribution,
        "duplicates": duplicate_images,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a labeled LeafCheck image batch and save its data report."
    )
    parser.add_argument("--batch-dir", default="dataset/new_batch")
    parser.add_argument("--classes-file", default="models/classes.json")
    parser.add_argument(
        "--report-path", default="monitoring/new_batch_data_report.json"
    )
    args = parser.parse_args()

    try:
        report = inspect_new_batch(args.batch_dir, args.classes_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    logger.info("Task A data report saved to %s", report_path)
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
