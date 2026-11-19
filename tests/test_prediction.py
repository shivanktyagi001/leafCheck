"""
LeafCheck - Prediction Unit Test Suite
Verifies image validation, normalization, and output contracts.
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.predict import (
    validate_image_bytes,
    preprocess_image,
    predict_leaf_disease,
    TARGET_IMAGE_SIZE,
    MAX_FILE_SIZE_BYTES
)
from src.disease_info import get_disease_details, DISEASE_KNOWLEDGE_BASE


def test_validate_image_bytes_valid():
    """Valid image byte stream returns PIL Image in RGB format."""
    buf = io.BytesIO()
    img = Image.new("RGBA", (100, 100), color=(50, 150, 50, 255))
    img.save(buf, format="PNG")
    bytes_data = buf.getvalue()

    result_img = validate_image_bytes(bytes_data)
    assert isinstance(result_img, Image.Image)
    assert result_img.mode == "RGB"


def test_validate_image_empty_raises():
    """Empty payload raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_image_bytes(b"")


def test_validate_image_oversized_raises():
    """Oversized payload raises ValueError."""
    with pytest.raises(ValueError, match="exceeds maximum"):
        validate_image_bytes(b"0" * (MAX_FILE_SIZE_BYTES + 10))


def test_preprocess_image_shape_and_range():
    """Preprocessed image must be (1, 224, 224, 3) and bounded in [-1.0, 1.0]."""
    test_img = Image.new("RGB", (500, 300), color=(128, 128, 128))
    tensor = preprocess_image(test_img)

    assert tensor.shape == (1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3)
    assert tensor.dtype == np.float32
    assert np.min(tensor) >= -1.05
    assert np.max(tensor) <= 1.05


def test_disease_knowledge_base_completeness():
    """All 10 tomato classes must have descriptions and treatments."""
    assert len(DISEASE_KNOWLEDGE_BASE) == 10
    for cls_name, info in DISEASE_KNOWLEDGE_BASE.items():
        assert "disease_name" in info
        assert "causal_agent" in info
        assert "symptoms" in info
        assert len(info["prevention"]) > 0
        assert len(info["treatment"]) > 0


def test_predict_leaf_disease_structure():
    """Integration test of prediction pipeline."""
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(40, 160, 40))
    img.save(buf, format="JPEG")

    res = predict_leaf_disease(buf.getvalue(), "sample_test.jpg")
    assert res.success is True
    assert res.filename == "sample_test.jpg"
    assert res.confidence >= 0.0 and res.confidence <= 1.0
    assert len(res.top_3_predictions) == 3
    assert res.inference_time_ms > 0
