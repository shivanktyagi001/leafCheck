import json

from PIL import Image

from src.prepare_new_data import inspect_new_batch


def _save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path, format="PNG")


def test_inspect_new_batch_reports_unique_class_distribution_and_duplicates(tmp_path):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["healthy", "diseased"]), encoding="utf-8")
    batch_dir = tmp_path / "new_batch"
    _save_image(batch_dir / "healthy" / "leaf-1.png", (0, 255, 0))
    _save_image(batch_dir / "healthy" / "leaf-copy.png", (0, 255, 0))
    _save_image(batch_dir / "diseased" / "leaf-2.png", (255, 0, 0))

    report = inspect_new_batch(str(batch_dir), str(classes_path))

    assert report["is_valid"] is True
    assert report["dataset_size"] == 3
    assert report["valid_image_count"] == 3
    assert report["unique_image_count"] == 2
    assert report["duplicate_count"] == 1
    assert report["class_distribution"] == {"healthy": 1, "diseased": 1}
    assert report["errors"] == []


def test_inspect_new_batch_flags_missing_and_unknown_labels(tmp_path):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["healthy"]), encoding="utf-8")
    batch_dir = tmp_path / "new_batch"
    batch_dir.mkdir()
    _save_image(batch_dir / "unrecognized" / "leaf.png", (1, 2, 3))
    _save_image(batch_dir / "unlabeled.png", (4, 5, 6))

    report = inspect_new_batch(str(batch_dir), str(classes_path))

    assert report["is_valid"] is False
    assert report["dataset_size"] == 2
    assert any("Unknown class label" in error for error in report["errors"])
    assert any("no class label" in error for error in report["errors"])


def test_inspect_new_batch_rejects_corrupt_images(tmp_path):
    classes_path = tmp_path / "classes.json"
    classes_path.write_text(json.dumps(["healthy"]), encoding="utf-8")
    image_path = tmp_path / "new_batch" / "healthy" / "broken.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not an image")

    report = inspect_new_batch(str(image_path.parents[1]), str(classes_path))

    assert report["is_valid"] is False
    assert report["dataset_size"] == 1
    assert report["valid_image_count"] == 0
    assert any("corrupt image" in error for error in report["errors"])
