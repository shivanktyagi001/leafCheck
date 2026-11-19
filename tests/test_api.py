"""
LeafCheck - API Automated Test Suite
Tests endpoints for health, classes, image prediction, and error handling.
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def create_test_image_bytes(format="JPEG", size=(224, 224), color=(60, 150, 45)) -> bytes:
    """Helper to generate in-memory dummy image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_root_endpoint():
    """Verify root / returns 200 and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "LeafCheck – Crop Disease Detection System"
    assert data["status"] == "online"


def test_health_endpoint():
    """Verify /health returns 200 and reports model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["classes_count"] == 10


def test_classes_endpoint():
    """Verify /classes returns 200 with 10 tomato categories."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 10
    assert "Tomato___Bacterial_spot" in data["classes"]
    assert "Tomato___healthy" in data["classes"]


def test_predict_valid_image():
    """Verify /predict correctly processes a valid leaf image."""
    img_bytes = create_test_image_bytes("JPEG")
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_class" in data
    assert "display_name" in data
    assert "confidence" in data
    assert len(data["top_3_predictions"]) == 3
    assert "disease_info" in data
    assert len(data["disease_info"]["prevention"]) > 0
    assert len(data["disease_info"]["treatment"]) > 0


def test_predict_corrupt_image():
    """Verify /predict returns 400 when uploaded bytes are not a valid image."""
    corrupt_bytes = b"This is definitely not an image file header."
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "not a valid or readable image" in response.json()["detail"].lower()


def test_predict_unsupported_content_type():
    """Verify /predict rejects disallowed MIME types."""
    dummy_text = b"Random plain text"
    files = {"file": ("notes.txt", dummy_text, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"].lower()


def test_predict_oversized_file():
    """Verify /predict rejects files over 10 MB with HTTP 413."""
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)  # 10MB + 1KB
    files = {"file": ("huge_leaf.jpg", oversized_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
