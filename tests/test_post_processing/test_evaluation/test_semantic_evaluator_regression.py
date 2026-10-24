"""
test_semantic_evaluator_regression.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[3] / "src"

# Expected results were produced by the evaluator before it became schema-driven
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


# Fuzzy key matching follows set iteration order, so the run uses the baseline's hash seed
RUNNER = """
import json, sys
from comproscanner.post_processing.evaluation.semantic_evaluator import MaterialsDataSemanticEvaluator
args = json.loads(sys.argv[1])
for key in ("init", "evaluate"):
    if "value_error_thresholds" in args[key]:
        thresholds = args[key]["value_error_thresholds"]
        args[key]["value_error_thresholds"] = {tuple(float(x) for x in k.split(",")): v for k, v in thresholds.items()}
evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=False, **args["init"])
result = evaluator.evaluate(extraction_agent_model_name="regression", **args["files"], **args["evaluate"])
"""


def _encode_thresholds(settings):
    """Encode tuple keys of value_error_thresholds for passing as JSON"""
    encoded = {}
    for key, values in settings.items():
        encoded[key] = dict(values)
        thresholds = values.get("value_error_thresholds")
        if thresholds:
            encoded[key]["value_error_thresholds"] = {
                f"{lo},{hi}": v for (lo, hi), v in thresholds.items()
            }
    return encoded


@pytest.mark.parametrize("case", sorted(FIXTURE))
def test_item_results_match_pre_schema_evaluator(case, tmp_path):
    data = FIXTURE[case]
    ground_truth_file = tmp_path / "ground_truth.json"
    test_data_file = tmp_path / "test_data.json"
    output_file = tmp_path / "result.json"
    ground_truth_file.write_text(json.dumps(data["ground_truth"]), encoding="utf-8")
    test_data_file.write_text(json.dumps(data["test_data"]), encoding="utf-8")

    args = _encode_thresholds(SETTINGS[case])
    args["files"] = {
        "ground_truth_file": str(ground_truth_file),
        "test_data_file": str(test_data_file),
        "output_file": str(output_file),
    }
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(SRC_DIR)}
    subprocess.run(
        [sys.executable, "-c", RUNNER, json.dumps(args)],
        env=env,
        check=True,
        capture_output=True,
    )
    result = json.loads(output_file.read_text(encoding="utf-8"))

    assert_close(result["item_results"], data["expected_item_results"])
