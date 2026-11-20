from PIL import Image

from src.train import build_data_pipelines


def _save_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path, format="PNG")


def test_candidate_pipeline_combines_batch_and_preserves_class_indices(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "validation"
    batch_dir = tmp_path / "new_batch"
    for split_dir in (train_dir, val_dir):
        _save_image(split_dir / "diseased" / "leaf.png", (255, 0, 0))
        _save_image(split_dir / "healthy" / "leaf.png", (0, 255, 0))
    _save_image(batch_dir / "healthy" / "new_leaf.png", (0, 200, 0))

    train_ds, _, classes = build_data_pipelines(
        {
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "img_height": 8,
            "img_width": 8,
        },
        batch_size=2,
        additional_train_dir=str(batch_dir),
    )

    label_counts = [0, 0]
    for _, labels in train_ds:
        for label in labels.numpy():
            label_counts[label.argmax()] += 1

    assert classes == ["diseased", "healthy"]
    assert label_counts == [1, 2]
