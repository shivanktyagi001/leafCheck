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
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import mlflow
import mlflow.keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def build_data_pipelines(
    data_cfg: Dict[str, Any],
    batch_size: int,
    additional_train_dir: Optional[str] = None,
):
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
        seed=42,
    )

    detected_classes = train_ds.class_names
    logger.info(f"Detected classes ({len(detected_classes)}): {detected_classes}")

    logger.info(f"Loading validation data from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    if additional_train_dir is not None:
        batch_root = Path(additional_train_dir)
        if not batch_root.is_dir():
            raise FileNotFoundError(f"New labeled batch directory not found: {batch_root}")

        unknown_classes = sorted(
            path.name
            for path in batch_root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name not in detected_classes
        )
        if unknown_classes:
            raise ValueError(f"New batch contains unknown classes: {unknown_classes}")

        batch_classes = sorted(
            path.name
            for path in batch_root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
        )
        if not batch_classes or not any(
            image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            for class_name in batch_classes
            for image in (batch_root / class_name).rglob("*")
        ):
            raise ValueError(f"No supported labeled images found in new batch: {batch_root}")

        batch_ds = tf.keras.utils.image_dataset_from_directory(
            str(batch_root),
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=batch_classes,
            shuffle=True,
            seed=42,
        )
        global_class_indices = tf.constant(
            [detected_classes.index(name) for name in batch_classes],
            dtype=tf.int32,
        )

        def map_batch_labels(images, labels):
            local_indices = tf.argmax(labels, axis=1, output_type=tf.int32)
            global_indices = tf.gather(global_class_indices, local_indices)
            return images, tf.one_hot(global_indices, depth=len(detected_classes))

        train_ds = train_ds.concatenate(batch_ds.map(map_batch_labels))
        train_ds = train_ds.shuffle(
            buffer_size=1000, seed=42, reshuffle_each_iteration=True
        )
        logger.info(
            "Added labeled batch %s with classes %s to the training data",
            batch_root,
            batch_classes,
        )

    # Performance optimization: prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, detected_classes


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


def train_pipeline(
    quick_run: bool = False,
    new_data_dir: Optional[str] = None,
    candidate_model_path: Optional[str] = None,
    candidate_classes_path: Optional[str] = None,
):
    """Executes the complete training workflow with MLflow tracking."""
    params = load_params()
    data_cfg = params["data"]
    model_cfg = params["model"]
    train_cfg = params["train"]
    mlflow_cfg = params["mlflow"]
    is_candidate = new_data_dir is not None

    if is_candidate:
        model_cfg = dict(model_cfg)
        model_cfg["model_path"] = candidate_model_path or (
            "models/candidate/leafcheck_candidate.keras"
        )
        model_cfg["classes_path"] = candidate_classes_path or (
            "models/candidate/classes.json"
        )
        candidate_paths = {
            "model": Path(model_cfg["model_path"]).resolve(),
            "class mapping": Path(model_cfg["classes_path"]).resolve(),
        }
        champion_paths = {
            "model": Path(
                params["model"].get("model_path", "models/leafcheck_model.keras")
            ).resolve(),
            "class mapping": Path(
                params["model"].get("classes_path", "models/classes.json")
            ).resolve(),
        }
        for candidate_name, candidate_path in candidate_paths.items():
            if candidate_path in champion_paths.values():
                raise ValueError(
                    f"Candidate {candidate_name} path must not overwrite a champion "
                    f"artifact: {candidate_path}"
                )
        if len(set(candidate_paths.values())) != len(candidate_paths):
            raise ValueError("Candidate model and class mapping paths must be different")

    batch_size = train_cfg.get("batch_size", 32)
    initial_epochs = 2 if quick_run else train_cfg.get("initial_epochs", 8)
    fine_tune_epochs = 2 if quick_run else train_cfg.get("fine_tune_epochs", 4)

    # Ensure output folders exist
    model_path = model_cfg.get("model_path", "models/leafcheck_model.keras")
    classes_path = model_cfg.get("classes_path", "models/classes.json")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(classes_path) or ".", exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "LeafCheck_Tomato_Disease"))

    run_name = "candidate_retraining" if is_candidate else "mobilenetv2_two_phase_training"
    with mlflow.start_run(run_name=run_name) as run:
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
            "image_size": f"{data_cfg.get('img_height', 224)}x{data_cfg.get('img_width', 224)}",
            "is_candidate": is_candidate,
        })
        if is_candidate:
            mlflow.log_param("new_labeled_data_dir", new_data_dir)
            mlflow.log_param(
                "combined_training_data_dirs",
                f"{data_cfg.get('train_dir', 'dataset/train')},{new_data_dir}",
            )

        # 2. Prepare Data
        train_ds, val_ds, class_names = build_data_pipelines(
            data_cfg,
            batch_size,
            additional_train_dir=new_data_dir,
        )

        # Save class mapping
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
    parser.add_argument(
        "--new-data-dir",
        help="Retrain a separate candidate using this labeled class-folder batch",
    )
    parser.add_argument(
        "--candidate-model-path",
        default="models/candidate/leafcheck_candidate.keras",
        help="Output path for the candidate model (does not replace the champion)",
    )
    parser.add_argument(
        "--candidate-classes-path",
        default="models/candidate/classes.json",
        help="Output path for the candidate class mapping",
    )
    args = parser.parse_args()
    train_pipeline(
        quick_run=args.quick,
        new_data_dir=args.new_data_dir,
        candidate_model_path=args.candidate_model_path if args.new_data_dir else None,
        candidate_classes_path=args.candidate_classes_path if args.new_data_dir else None,
    )
