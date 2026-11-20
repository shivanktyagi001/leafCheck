"""Run the labeled-batch validation, evaluation, retraining, and quality gate."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("leafcheck.workflow")


def quality_gate(
    champion_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    minimum_improvement: float = 0.0,
) -> dict[str, Any]:
    """Approve a candidate only when accuracy and weighted F1 do not regress."""
    required_metrics = ("accuracy", "f1")
    for label, metrics in (("champion", champion_metrics), ("candidate", candidate_metrics)):
        missing = [metric for metric in required_metrics if metric not in metrics]
        if missing:
            raise ValueError(f"{label} metrics are missing: {missing}")

    accuracy_improvement = candidate_metrics["accuracy"] - champion_metrics["accuracy"]
    f1_improvement = candidate_metrics["f1"] - champion_metrics["f1"]
    approved = (
        accuracy_improvement >= minimum_improvement
        and f1_improvement >= minimum_improvement
    )
    return {
        "status": "approved" if approved else "rejected",
        "minimum_improvement": minimum_improvement,
        "accuracy_improvement": accuracy_improvement,
        "f1_improvement": f1_improvement,
        "reason": (
            "Candidate meets the accuracy and weighted F1 quality gates."
            if approved
            else "Candidate does not meet the accuracy and weighted F1 quality gates."
        ),
    }


def run_workflow(
    batch_dir: str = "dataset/new_batch",
    champion_model_path: str = "models/leafcheck_model.keras",
    champion_classes_path: str = "models/classes.json",
    output_dir: str = "monitoring/workflow",
    candidate_model_path: str = "models/candidate/leafcheck_candidate.keras",
    candidate_classes_path: str = "models/candidate/classes.json",
    quick_run: bool = False,
    minimum_improvement: float = 0.0,
) -> dict[str, Any]:
    """Execute the complete workflow and persist its result as JSON."""
    from src.evaluate import evaluate_test_dataset
    from src.prepare_new_data import inspect_new_batch
    from src.train import train_pipeline

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_report = inspect_new_batch(batch_dir, champion_classes_path)
    if not data_report["is_valid"]:
        raise ValueError(f"Incoming batch failed validation: {data_report['errors']}")

    champion_report = evaluate_test_dataset(
        model_path=champion_model_path,
        classes_path=champion_classes_path,
        test_dir=batch_dir,
        output_dir=str(output_path / "champion"),
    )
    train_pipeline(
        quick_run=quick_run,
        new_data_dir=batch_dir,
        candidate_model_path=candidate_model_path,
        candidate_classes_path=candidate_classes_path,
    )
    candidate_report = evaluate_test_dataset(
        model_path=candidate_model_path,
        classes_path=candidate_classes_path,
        test_dir=batch_dir,
        output_dir=str(output_path / "candidate"),
    )
    decision = quality_gate(
        champion_report,
        candidate_report,
        minimum_improvement=minimum_improvement,
    )
    result = {
        "workflow_status": "successful",
        "steps": [
            "validated labeled batch",
            "evaluated champion model",
            "trained candidate model",
            "evaluated candidate model",
            "applied quality gate",
        ],
        "data_report": data_report,
        "champion_evaluation": champion_report,
        "candidate_evaluation": candidate_report,
        "candidate_artifacts": {
            "model": candidate_model_path,
            "classes": candidate_classes_path,
        },
        "quality_gate": decision,
    }
    result_path = output_path / "workflow_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete LeafCheck model workflow.")
    parser.add_argument("--batch-dir", default="dataset/new_batch")
    parser.add_argument("--champion-model-path", default="models/leafcheck_model.keras")
    parser.add_argument("--champion-classes-path", default="models/classes.json")
    parser.add_argument("--candidate-model-path", default="models/candidate/leafcheck_candidate.keras")
    parser.add_argument("--candidate-classes-path", default="models/candidate/classes.json")
    parser.add_argument("--output-dir", default="monitoring/workflow")
    parser.add_argument("--quick", action="store_true", help="Use short training epochs")
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    args = parser.parse_args()
    result = run_workflow(
        batch_dir=args.batch_dir,
        champion_model_path=args.champion_model_path,
        champion_classes_path=args.champion_classes_path,
        output_dir=args.output_dir,
        candidate_model_path=args.candidate_model_path,
        candidate_classes_path=args.candidate_classes_path,
        quick_run=args.quick,
        minimum_improvement=args.minimum_improvement,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["quality_gate"]["status"] == "approved" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
