"""
LeafCheck - Prediction Service
Performs image validation, exact training preprocessing, model inference,
top-3 candidate ranking, and disease knowledge retrieval.
"""

import io
import time
import logging
from typing import Tuple, List, Dict, Any
from PIL import Image, UnidentifiedImageError
import numpy as np

from src.model_loader import get_model, load_classes
from src.disease_info import get_disease_details
from src.schemas import PredictionResponse, PredictionCandidate, DiseaseDetails

logger = logging.getLogger("leafcheck.predict")

# Constants
TARGET_IMAGE_SIZE = (224, 224)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


def validate_image_bytes(file_bytes: bytes, content_type: str = None) -> Image.Image:
    """
    Validates file size, content type, and integrity.
    Returns opened PIL Image in RGB format.
    """
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")
    
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of 10 MB."
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify image integrity
        # Re-open after verify() as recommended by PIL
        image = Image.open(io.BytesIO(file_bytes))
    except (UnidentifiedImageError, OSError, Exception) as e:
        raise ValueError(f"Uploaded file is not a valid or readable image. Error: {str(e)}")

    if image.format and image.format.lower() not in ["jpeg", "jpg", "png", "webp"]:
        raise ValueError(
            f"Unsupported image format '{image.format}'. Supported formats: JPEG, PNG, WEBP."
        )

    # Convert to standard RGB (handles RGBA, grayscale, paletted)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resizes image to (224, 224) and applies MobileNetV2 preprocessing:
    Scales pixel values from [0, 255] to [-1.0, 1.0].
    """
    # High-quality Lanczos resampling
    resized_img = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)
    
    # MobileNetV2 preprocess formula: (x / 127.5) - 1.0
    preprocessed = (img_array / 127.5) - 1.0
    
    # Add batch dimension: shape (1, 224, 224, 3)
    batch_img = np.expand_dims(preprocessed, axis=0)
    return batch_img


def predict_leaf_disease(file_bytes: bytes, filename: str = "upload.jpg") -> PredictionResponse:
    """
    Complete inference workflow:
    1. Validation
    2. Preprocessing
    3. Inference
    4. Top-3 calculation
    5. Knowledge base enrichment
    """
    # 1. Validation
    image = validate_image_bytes(file_bytes)

    # 2. Preprocessing
    preprocessed_tensor = preprocess_image(image)

    # 3. Model execution
    model = get_model()
    classes = load_classes()

    start_time = time.perf_counter()
    raw_predictions = model.predict(preprocessed_tensor, verbose=0)
    inference_time = (time.perf_counter() - start_time) * 1000.0  # in ms

    probabilities = raw_predictions[0]

    # Handle cases where model outputs raw logits instead of softmax
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-2):
        exp_preds = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_preds / np.sum(exp_preds)

    # 4. Top-3 Extraction
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_candidates: List[PredictionCandidate] = []

    for idx in top_indices:
        cls_name = classes[idx] if idx < len(classes) else f"Class_{idx}"
        conf = float(probabilities[idx])
        info = get_disease_details(cls_name)
        top_candidates.append(
            PredictionCandidate(
                class_name=cls_name,
                display_name=info["disease_name"],
                confidence=round(conf, 4),
                percentage=f"{conf * 100.0:.2f}%"
            )
        )

    best_candidate = top_candidates[0]
    best_disease_info = get_disease_details(best_candidate.class_name)

    # Production Confidence Thresholding (OOD detection)
    CONFIDENCE_THRESHOLD = 0.60
    is_confident = best_candidate.confidence >= CONFIDENCE_THRESHOLD
    warning_msg = None
    if not is_confident:
        warning_msg = (
            f"Low confidence warning ({best_candidate.percentage} < {int(CONFIDENCE_THRESHOLD*100)}%). "
            "The model is uncertain. The uploaded file may be out-of-distribution (e.g. not a leaf, "
            "contains cluttered background, or symptoms are ambiguous). Please upload a clear closeup of a single tomato leaf."
        )

    return PredictionResponse(
        success=True,
        filename=filename,
        predicted_class=best_candidate.class_name,
        display_name=best_candidate.display_name,
        confidence=best_candidate.confidence,
        confidence_percentage=best_candidate.percentage,
        is_confident=is_confident,
        warning=warning_msg,
        top_3_predictions=top_candidates,
        disease_info=DiseaseDetails(**best_disease_info),
        inference_time_ms=round(inference_time, 2)
    )
