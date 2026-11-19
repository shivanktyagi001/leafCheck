"""
LeafCheck - Model Loader Singleton
Responsible for loading and caching the Keras model and class labels.
Prevents reloading the heavy model on every HTTP request.
"""

import os
import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("leafcheck.model_loader")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join("models", "leafcheck_model.keras")
CLASSES_PATH = os.path.join("models", "classes.json")

# In-memory cached instances
_MODEL = None
_CLASSES: Optional[List[str]] = None


def load_classes(classes_path: str = CLASSES_PATH) -> List[str]:
    """Load class labels from JSON."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            _CLASSES = json.load(f)
        logger.info(f"Loaded {len(_CLASSES)} classes from {classes_path}")
    else:
        # Default canonical 10 tomato classes if file missing
        _CLASSES = [
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
        logger.warning(f"Classes file {classes_path} not found. Used fallback default 10 classes.")
    return _CLASSES


def get_model(model_path: str = MODEL_PATH):
    """Load and return the trained Keras model, caching it in memory."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(model_path):
        logger.warning(f"Trained model not found at '{model_path}'. Initializing base MobileNetV2 for inference readiness.")
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Build base architecture to ensure inference pipeline can function
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(10, activation="softmax")(x)
        _MODEL = models.Model(inputs=base.input, outputs=outputs, name="LeafCheck_MobileNetV2_Unfitted")
        return _MODEL

    logger.info(f"Loading trained Keras model from {model_path}...")
    import tensorflow as tf
    _MODEL = tf.keras.models.load_model(model_path)
    logger.info("Keras model loaded successfully.")
    return _MODEL


def get_model_and_classes() -> Tuple[any, List[str]]:
    """Convenience getter for both model and classes."""
    return get_model(), load_classes()
