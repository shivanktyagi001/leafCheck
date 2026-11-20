"""
LeafCheck - Model & Data Drift Monitoring with Evidently AI
Monitors:
1. Prediction Distribution Drift (Target Drift)
2. Confidence Score Degradation
3. Model Performance Drift over time
Produces HTML reports and structured JSON summaries.
"""

import os

os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leafcheck.monitoring")

REQUIRED_COLUMNS = {"predicted_class", "confidence", "latency_ms"}


def _load_data(path: str) -> pd.DataFrame:
    """Load a monitoring dataset from CSV or JSON and validate its schema."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Monitoring data file does not exist: {data_path}")

    if data_path.suffix.lower() == ".csv":
        data = pd.read_csv(data_path)
    elif data_path.suffix.lower() == ".json":
        data = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported monitoring data format: {data_path.suffix}")

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Monitoring data {data_path} is missing required columns: {missing_columns}"
        )
    if data.empty:
        raise ValueError(f"Monitoring data {data_path} contains no rows")

    for column in ("confidence", "latency_ms"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["confidence", "latency_ms"]].isna().any().any():
        raise ValueError(f"Monitoring data {data_path} contains non-numeric metrics")
    if ((data["confidence"] < 0) | (data["confidence"] > 1)).any():
        raise ValueError(f"Confidence values in {data_path} must be between 0 and 1")
    return data


def _demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic data for the documented no-argument demo command."""
    np.random.seed(42)
    classes = [
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]
    reference = pd.DataFrame({
        "predicted_class": np.random.choice(classes, size=300),
        "confidence": np.random.beta(a=12, b=2, size=300),
        "latency_ms": np.random.normal(loc=45, scale=8, size=300),
    })
    current = pd.DataFrame({
        "predicted_class": np.random.choice(classes, size=150),
        "confidence": np.random.beta(a=8, b=2.5, size=150),
        "latency_ms": np.random.normal(loc=52, scale=12, size=150),
    })
    return reference, current


def _summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Return stable operational summaries independent of Evidently's API version."""
    return {
        "reference_sample_count": len(reference),
        "current_sample_count": len(current),
        "reference": {
            "mean_confidence": float(reference["confidence"].mean()),
            "mean_latency_ms": float(reference["latency_ms"].mean()),
            "class_distribution": reference["predicted_class"].value_counts(normalize=True).to_dict(),
        },
        "current": {
            "mean_confidence": float(current["confidence"].mean()),
            "mean_latency_ms": float(current["latency_ms"].mean()),
            "class_distribution": current["predicted_class"].value_counts(normalize=True).to_dict(),
        },
    }


def generate_drift_monitoring_report(
    reference_data_path: str | None = None,
    current_data_path: str | None = None,
    output_html_path: str = "monitoring/drift_report.html",
    summary_json_path: str = "monitoring/drift_summary.json",
) -> str:
    """
    Compares baseline validation distribution with incoming production inference requests.
    Generates an Evidently HTML dashboard and prints drift metrics.
    """
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(summary_json_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if reference_data_path and current_data_path:
        df_ref = _load_data(reference_data_path)
        df_cur = _load_data(current_data_path)
    elif reference_data_path or current_data_path:
        raise ValueError("Both reference_data_path and current_data_path are required")
    else:
        logger.warning("No monitoring data supplied; generating the documented demo dataset")
        df_ref, df_cur = _demo_data()

    summary = _summary(df_ref, df_cur)
    summary["reference_data_path"] = reference_data_path
    summary["current_data_path"] = current_data_path
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        from evidently.report import Report
        from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required to generate the HTML drift report. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    logger.info("Computing drift metrics across features and target distributions...")
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="confidence"),
        ColumnDriftMetric(column_name="predicted_class"),
        ColumnDriftMetric(column_name="latency_ms")
    ])

    report.run(reference_data=df_ref, current_data=df_cur)

    # 4. Save HTML Report
    report.save_html(str(output_path))
    logger.info("Evidently drift report saved to %s", output_path)
    logger.info("Drift summary saved to %s", summary_path)

    # Extract summary dictionary
    summary_dict = report.as_dict()
    metrics = summary_dict.get("metrics", [])
    for m in metrics:
        metric_name = m.get("metric")
        result = m.get("result", {})
        drift_detected = result.get("dataset_drift") or result.get("drift_detected")
        logger.info(f"Evidently Metric [{metric_name}]: Drift Detected = {drift_detected}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a LeafCheck drift monitoring report.")
    parser.add_argument("--reference-data", help="Reference CSV or JSON monitoring dataset")
    parser.add_argument("--current-data", help="Current CSV or JSON monitoring dataset")
    parser.add_argument("--output-html", default="monitoring/drift_report.html")
    parser.add_argument("--summary-json", default="monitoring/drift_summary.json")
    args = parser.parse_args()
    generate_drift_monitoring_report(
        reference_data_path=args.reference_data,
        current_data_path=args.current_data,
        output_html_path=args.output_html,
        summary_json_path=args.summary_json,
    )
