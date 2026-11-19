"""
LeafCheck - Model Training Pipeline with Transfer Learning & MLflow
Implements two-stage training:
1. Feature extraction with frozen MobileNetV2 base
2. Fine-tuning of top layers with low learning rate
Logs all parameters, metrics, and artifacts to MLflow.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import sys
import json
import yaml
import logging
import argparse
from typing import Dict, Any, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import mlflow
import mlflow.keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.train")


def load_params(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    # Default fallback parameters
    return {
        "data": {
            "train_dir": "dataset/train",
            "val_dir": "dataset/validation",
            "test_dir": "dataset/test",
            "img_height": 224,
            "img_width": 224
        },
        "model": {
            "architecture": "MobileNetV2",
            "dropout_rate": 0.3,
            "num_classes": 10,
            "model_path": "models/leafcheck_model.keras",
            "classes_path": "models/classes.json"
        },
        "train": {
            "batch_size": 32,
            "initial_epochs": 8,
            "initial_lr": 0.001,
            "fine_tune_epochs": 4,
            "fine_tune_lr": 0.00005,
            "fine_tune_layers": 20
        },
        "mlflow": {
            "experiment_name": "LeafCheck_Tomato_Disease"
        }
    }


def build_data_pipelines(data_cfg: Dict[str, Any], batch_size: int):
    """Creates memory-efficient tf.data input pipelines."""
    img_size = (data_cfg.get("img_height", 224), data_cfg.get("img_width", 224))
    train_dir = data_cfg.get("train_dir", "dataset/train")
    val_dir = data_cfg.get("val_dir", "dataset/validation")

    logger.info(f"Loading training data from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    class_names = train_ds.class_names
    logger.info(f"Detected classes ({len(class_names)}): {class_names}")

    logger.info(f"Loading validation data from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    # Performance optimization: prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(model_cfg: Dict[str, Any], img_shape=(224, 224, 3)) -> Tuple[models.Model, models.Model]:
    """
    Constructs Transfer Learning Model using MobileNetV2:
    Data Augmentation -> Preprocessing -> Base -> GlobalAvgPool -> Dropout -> Dense(Softmax)
    """
    num_classes = model_cfg.get("num_classes", 10)
    dropout_rate = model_cfg.get("dropout_rate", 0.3)

    inputs = layers.Input(shape=img_shape, name="input_leaf_image")

    # 1. On-device Data Augmentation (active only during training)
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.12, name="aug_rotation")(x)
    x = layers.RandomZoom(0.10, name="aug_zoom")(x)

    # 2. Preprocessing: MobileNetV2 normalization to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 3. Pretrained Feature Extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze initially

    x = base_model(x, training=False)

    # 4. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LeafCheck_MobileNetV2")
    return model, base_model


def train_pipeline(quick_run: bool = False):
    """Executes the complete training workflow with MLflow tracking."""
    params = load_params()
    data_cfg = params["data"]
    model_cfg = params["model"]
    train_cfg = params["train"]
    mlflow_cfg = params["mlflow"]

    batch_size = train_cfg.get("batch_size", 32)
    initial_epochs = 2 if quick_run else train_cfg.get("initial_epochs", 8)
    fine_tune_epochs = 2 if quick_run else train_cfg.get("fine_tune_epochs", 4)

    # Ensure output folders exist
    os.makedirs(model_cfg.get("model_dir", "models"), exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "LeafCheck_Tomato_Disease"))

    with mlflow.start_run(run_name="mobilenetv2_two_phase_training") as run:
        logger.info(f"Started MLflow Run ID: {run.info.run_id}")

        # 1. Log hyperparameters
        mlflow.log_params({
            "architecture": model_cfg.get("architecture", "MobileNetV2"),
            "initial_epochs": initial_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "batch_size": batch_size,
            "initial_lr": train_cfg.get("initial_lr", 0.001),
            "fine_tune_lr": train_cfg.get("fine_tune_lr", 0.00005),
            "fine_tune_layers": train_cfg.get("fine_tune_layers", 20),
            "dropout_rate": model_cfg.get("dropout_rate", 0.3),
            "image_size": f"{data_cfg.get('img_height', 224)}x{data_cfg.get('img_width', 224)}"
        })

        # 2. Prepare Data
        train_ds, val_ds, class_names = build_data_pipelines(data_cfg, batch_size)

        # Save class mapping
        classes_path = model_cfg.get("classes_path", "models/classes.json")
        with open(classes_path, "w") as f:
            json.dump(class_names, f, indent=2)
        mlflow.log_artifact(classes_path)

        # 3. Build Model
        model, base_model = build_model(model_cfg)
        model.summary(print_fn=logger.info)

        # 4. Phase 1: Feature Extraction (Frozen Base)
        logger.info("=== STAGE 1: Training Classification Head (Base Model Frozen) ===")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=train_cfg.get("initial_lr", 0.001)),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
        )

        model_path = model_cfg.get("model_path", "models/leafcheck_model.keras")
        cb_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(filepath=model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
        ]

        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs,
            callbacks=cb_list
        )

        for ep in range(len(history_phase1.history["accuracy"])):
            mlflow.log_metric("train_acc", history_phase1.history["accuracy"][ep], step=ep + 1)
            mlflow.log_metric("val_acc", history_phase1.history["val_accuracy"][ep], step=ep + 1)
            mlflow.log_metric("train_loss", history_phase1.history["loss"][ep], step=ep + 1)
            mlflow.log_metric("val_loss", history_phase1.history["val_loss"][ep], step=ep + 1)

        # 5. Phase 2: Fine-Tuning Top Layers
        logger.info("=== STAGE 2: Fine-Tuning Top MobileNetV2 Layers ===")
        base_model.trainable = True
        fine_tune_layers = train_cfg.get("fine_tune_layers", 20)
        # Freeze all layers except the last N
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=train_cfg.get("fine_tune_lr", 0.00005)),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
        )

        total_epochs = initial_epochs + fine_tune_epochs
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=total_epochs,
            callbacks=cb_list
        )

        for ep in range(len(history_phase2.history["accuracy"])):
            step = initial_epochs + ep + 1
            mlflow.log_metric("train_acc", history_phase2.history["accuracy"][ep], step=step)
            mlflow.log_metric("val_acc", history_phase2.history["val_accuracy"][ep], step=step)
            mlflow.log_metric("train_loss", history_phase2.history["loss"][ep], step=step)
            mlflow.log_metric("val_loss", history_phase2.history["val_loss"][ep], step=step)

        # Save final model artifact to MLflow
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
            logger.info(f"Model successfully saved and logged to MLflow: {model_path}")

        logger.info("Training pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeafCheck MobileNetV2 model")
    parser.add_argument("--quick", action="store_true", help="Run short training for fast verification")
    args = parser.parse_args()
    train_pipeline(quick_run=args.quick)
