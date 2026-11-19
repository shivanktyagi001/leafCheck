"""
LeafCheck - Model Evaluation Module
Evaluates model on held-out test dataset.
Generates:
1. Confusion Matrix plot
2. Precision, Recall, F1-Score classification report
3. MLflow evaluation metrics logging
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import json
import logging
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.evaluate")


def evaluate_test_dataset(
    model_path: str = "models/leafcheck_model.keras",
    classes_path: str = "models/classes.json",
    test_dir: str = "dataset/test",
    batch_size: int = 32,
    img_size: tuple = (224, 224),
    output_dir: str = "monitoring"
) -> Dict[str, Any]:
    """Runs complete evaluation against the unseen test split."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")

    # 1. Load classes
    with open(classes_path, "r") as f:
        classes = json.load(f)

    # 2. Load model
    logger.info(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 3. Load test dataset
    logger.info(f"Loading test images from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    # 4. Collect true labels and predictions
    y_true_indices = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true_indices.append(np.argmax(labels.numpy(), axis=1))

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_true_indices = np.concatenate(y_true_indices, axis=0)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # 5. Compute metrics
    acc = float(accuracy_score(y_true_indices, y_pred_indices))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="weighted", zero_division=0
    )

    logger.info(f"Test Accuracy : {acc:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall   : {recall:.4f}")
    logger.info(f"Test F1-Score : {f1:.4f}")

    # 6. Classification Report
    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # 7. Confusion Matrix Plot
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("Tomato___", "") for c in classes],
        yticklabels=[c.replace("Tomato___", "") for c in classes]
    )
    plt.title("LeafCheck - Confusion Matrix on Unseen Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_path}")

    # 8. Log metrics into MLflow if active run exists or start new eval run
    try:
        with mlflow.start_run(run_name="model_test_evaluation"):
            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_precision": float(precision),
                "test_recall": float(recall),
                "test_f1_score": float(f1)
            })
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(cm_path)
    except Exception as me:
        logger.warning(f"Could not log to MLflow: {me}")

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report_path": report_path,
        "confusion_matrix_path": cm_path
    }


if __name__ == "__main__":
    evaluate_test_dataset()
