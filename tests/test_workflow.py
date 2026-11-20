import pytest

from src.run_workflow import quality_gate


def test_quality_gate_approves_candidate_without_regression():
    decision = quality_gate(
        {"accuracy": 0.80, "f1": 0.75},
        {"accuracy": 0.81, "f1": 0.75},
    )

    assert decision["status"] == "approved"


def test_quality_gate_rejects_candidate_regression():
    decision = quality_gate(
        {"accuracy": 0.80, "f1": 0.75},
        {"accuracy": 0.81, "f1": 0.74},
    )

    assert decision["status"] == "rejected"


def test_quality_gate_requires_metrics():
    with pytest.raises(ValueError, match="missing"):
        quality_gate({"accuracy": 0.8}, {"accuracy": 0.8, "f1": 0.8})
