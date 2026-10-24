"""
test_eval_flow_regression.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

# Import the actual class before conftest runs its mocks
import sys
import os

if "PYTEST_CURRENT_TEST" in os.environ:
    crewai_mocks = [key for key in sys.modules.keys() if key.startswith("crewai")]
    saved_mocks = {
        key: sys.modules.pop(key) for key in crewai_mocks if key in sys.modules
    }

import copy
import json
from pathlib import Path

import pytest

from comproscanner.post_processing.evaluation.eval_flow.eval_flow import (
    MaterialsDataAgenticEvaluatorFlow,
)

# Recorded agent decisions and the scores the evaluator produced before it became schema-driven
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "agentic_evaluator_regression.json").read_text(
        encoding="utf-8"
    )
)
METRICS = ("true_positives", "false_positives", "false_negatives")
SETTINGS = {
    "deepseek-v3-0324": {},
    "custom_weights": {
        "weights": {
            "compositions_property_values": 0.4,
            "property_unit": 0.05,
            "family": 0.05,
            "method": 0.2,
            "precursors": 0.1,
            "characterization_techniques": 0.1,
            "steps": 0.1,
        }
    },
    "composition_only": {"is_synthesis_evaluation": False},
}


@pytest.fixture
def data_files(tmp_path):
    ground_truth = tmp_path / "ground_truth.json"
    test_data = tmp_path / "test_data.json"
    ground_truth.write_text("{}", encoding="utf-8")
    test_data.write_text("{}", encoding="utf-8")
    return str(ground_truth), str(test_data)


def score_item(flow, recorded):
    """Score one paper's recorded agent decisions the way evaluate_items and finalize_evaluation do."""
    sections = ["composition_data"]
    if flow.state.is_synthesis_evaluation:
        sections.append("synthesis_data")
    details = {
        section: copy.deepcopy(recorded[section])
        for section in sections
        if recorded.get(section)
    }
    result = {
        "field_scores": {},
        "absolute_classification_metrics": dict.fromkeys(METRICS, 0),
        "normalized_classification_metrics": dict.fromkeys(METRICS, 0.0),
    }
    if not any(not flow._is_section_empty(details[s]) for s in details):
        return result

    enhanced = flow._enhance_evaluation_details(details)
    normalized_weights = flow._normalize_weights()
    for section in sections:
        if section in enhanced and not flow._is_section_empty(enhanced[section]):
            result["field_scores"][section] = flow._calculate_score(
                enhanced, section, flow.state.weights
            )
            metrics = flow._calculate_tp_fp_fn(enhanced, section)
            for metric in METRICS:
                result["absolute_classification_metrics"][metric] += metrics[metric]
    # Normalized metrics as recomputed by finalize_evaluation
    final_details = flow._enhance_evaluation_details(enhanced)
    for section in sections:
        metrics = flow._calculate_normalized_metrics(
            final_details, section, normalized_weights
        )
        for metric in METRICS:
            result["normalized_classification_metrics"][metric] += metrics[metric]
    return result


@pytest.mark.parametrize("run", sorted(SETTINGS))
def test_scores_match_pre_schema_evaluator(run, data_files):
    ground_truth_file, test_data_file = data_files
    flow = MaterialsDataAgenticEvaluatorFlow(
        ground_truth_file=ground_truth_file,
        test_data_file=test_data_file,
        extraction_agent_model_name="regression",
        **SETTINGS[run],
    )
    for doi, expected in FIXTURE["expected"][run].items():
        actual = score_item(flow, FIXTURE["recorded_details"][doi])
        assert actual["field_scores"] == pytest.approx(expected["field_scores"], abs=1e-12), doi
        for kind in ("absolute_classification_metrics", "normalized_classification_metrics"):
            for metric in METRICS:
                assert actual[kind][metric] == pytest.approx(
                    expected[kind][metric], abs=1e-12
                ), (doi, kind, metric)
