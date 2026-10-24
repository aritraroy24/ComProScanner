"""
test_semantic_evaluator_regression.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

import json
from pathlib import Path

import pytest

from comproscanner.post_processing.evaluation.semantic_evaluator import (
    MaterialsDataSemanticEvaluator,
)

# Expected results: the evaluator before it became schema-driven, updated for deterministic key order and swap correction
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "semantic_evaluator_regression.json").read_text(
        encoding="utf-8"
    )
)

VLM_SETTINGS = {
    "init": {
        "similarity_thresholds": {"compositions_property_values": 1.0},
        "value_error_thresholds": {
            (-50, 50): 0.5,
            (-150, 150): 1,
            (float("-inf"), float("inf")): 2,
        },
    },
    "evaluate": {
        "is_synthesis_evaluation": False,
        "weights": {
            "compositions_property_values": 1.0,
            "property_unit": 0.0,
            "family": 0.0,
            "method": 0.0,
            "precursors": 0.0,
            "characterization_techniques": 0.0,
            "steps": 0.0,
        },
    },
}
SETTINGS = {
    "piezo_default": {"init": {}, "evaluate": {}},
    "piezo_missing_dois": {"init": {}, "evaluate": {}},
    "vlm_settings": VLM_SETTINGS,
}


def assert_close(actual, expected, path=""):
    """Assert nested results are equal, allowing float differences from summation order."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), f"{path}: {set(actual) ^ set(expected)}"
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=0, abs=1e-12), path
    else:
        assert actual == expected, path


@pytest.mark.parametrize("case", sorted(FIXTURE))
def test_item_results_match_reference(case, tmp_path):
    data = FIXTURE[case]
    settings = SETTINGS[case]
    ground_truth_file = tmp_path / "ground_truth.json"
    test_data_file = tmp_path / "test_data.json"
    ground_truth_file.write_text(json.dumps(data["ground_truth"]), encoding="utf-8")
    test_data_file.write_text(json.dumps(data["test_data"]), encoding="utf-8")

    evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=False, **settings["init"])
    result = evaluator.evaluate(
        ground_truth_file=str(ground_truth_file),
        test_data_file=str(test_data_file),
        extraction_agent_model_name="regression",
        output_file=str(tmp_path / "result.json"),
        **settings["evaluate"],
    )

    assert_close(result["item_results"], data["expected_item_results"])
