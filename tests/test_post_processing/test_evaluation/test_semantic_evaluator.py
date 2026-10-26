"""
test_semantic_evaluator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-07-2025
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
from comproscanner.post_processing.evaluation.semantic_evaluator import (
    MaterialsDataSemanticEvaluator,
)
from comproscanner.utils.error_handler import ValueErrorHandler, BaseError


class TestMaterialsDataSemanticEvaluator:
    """Test suite for MaterialsDataSemanticEvaluator class"""

    @pytest.fixture
    def evaluator_no_model(self):
        """Fixture for evaluator without semantic models"""
        return MaterialsDataSemanticEvaluator(use_semantic_model=False)

    @pytest.fixture
    def evaluator_with_model(self):
        """Fixture for evaluator with mocked semantic models"""
        with (
            patch("transformers.AutoModel.from_pretrained") as mock_model,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("torch.no_grad"),
            patch("torch.nn.functional.cosine_similarity") as mock_cosine,
        ):

            mock_cosine.return_value.item.return_value = 0.85
            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)
            return evaluator

    @pytest.fixture
    def sample_composition_data(self):
        """Sample composition data for testing"""
        return {
            "property_unit": "MPa",
            "family": "ceramics",
            "compositions_property_values": {
                "Al2O3": 150.5,
                "SiO2": [100, 200],
                "TiO2": 75.2,
            },
        }

    @pytest.fixture
    def sample_synthesis_data(self):
        """Sample synthesis data for testing"""
        return {
            "method": "sol-gel synthesis",
            "precursors": ["aluminum nitrate", "silicon oxide", "titanium dioxide"],
            "characterization_techniques": ["XRD", "SEM", "TEM"],
            "steps": ["Mix precursors", "Heat at 500C", "Cool slowly", "Analyze"],
        }

    @pytest.fixture
    def sample_ground_truth_data(self, sample_composition_data, sample_synthesis_data):
        """Sample ground truth data"""
        return {
            "10.1000/test1": {
                "composition_data": sample_composition_data,
                "synthesis_data": sample_synthesis_data,
            },
            "10.1000/test2": {
                "composition_data": {
                    "property_unit": "GPa",
                    "family": "polymers",
                    "compositions_property_values": {"PE": 2.5, "PP": 3.0},
                },
                "synthesis_data": {
                    "method": "polymerization",
                    "precursors": ["ethylene", "propylene"],
                    "characterization_techniques": ["FTIR", "DSC"],
                    "steps": ["Polymerize", "Purify"],
                },
            },
        }

    @pytest.fixture
    def sample_test_data(self, sample_composition_data, sample_synthesis_data):
        """Sample test data (similar to ground truth for positive testing)"""
        return {
            "10.1000/test1": {
                "composition_data": sample_composition_data.copy(),
                "synthesis_data": sample_synthesis_data.copy(),
            },
            "10.1000/test2": {
                "composition_data": {
                    "property_unit": "GPa",
                    "family": "polymers",
                    "compositions_property_values": {
                        "polyethylene": 2.4,  # Slightly different value
                        "polypropylene": 3.1,  # Slightly different value
                    },
                },
                "synthesis_data": {
                    "method": "radical polymerization",  # Similar but different
                    "precursors": ["ethylene monomer", "propylene monomer"],  # Similar
                    "characterization_techniques": [
                        "FTIR spectroscopy",
                        "DSC analysis",
                    ],  # Similar
                    "steps": ["Initiate polymerization", "Purify product"],  # Similar
                },
            },
        }

    @pytest.fixture
    def temp_json_files(self, sample_ground_truth_data, sample_test_data):
        """Create temporary JSON files for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            with open(gt_file, "w") as f:
                json.dump(sample_ground_truth_data, f)

            with open(test_file, "w") as f:
                json.dump(sample_test_data, f)

            yield gt_file, test_file, output_file

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=False)

        assert evaluator.use_semantic_model is False
        assert evaluator.primary_model_name == "thellert/physbert_cased"
        assert evaluator.fallback_model_name == "all-mpnet-base-v2"
        assert evaluator.physbert_available is False
        assert evaluator.model_available is False
        assert "composition_overall_match" in evaluator.similarity_thresholds
        assert "synthesis_overall_match" in evaluator.similarity_thresholds

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_thresholds = {"composition_overall_match": 0.75}
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False,
            primary_model_name="custom/model",
            fallback_model_name="custom/fallback",
            similarity_thresholds=custom_thresholds,
        )

        assert evaluator.primary_model_name == "custom/model"
        assert evaluator.fallback_model_name == "custom/fallback"
        assert evaluator.similarity_thresholds["composition_overall_match"] == 0.75

    def test_simple_preprocess(self, evaluator_no_model):
        """Test text preprocessing functionality"""
        text = "This is a TEST text with PUNCTUATION!"
        processed = evaluator_no_model._simple_preprocess(text)

        # Should be lowercase and without punctuation
        assert processed.islower()
        assert "!" not in processed
        assert "test" in processed

    def test_is_value_in_range_exact_match(self, evaluator_no_model):
        """Test exact value matching"""
        assert evaluator_no_model._is_value_in_range(10.5, 10.5) is True
        assert evaluator_no_model._is_value_in_range(10, 10.0) is True
        assert evaluator_no_model._is_value_in_range(10.5, 11.0) is False

    def test_is_value_in_range_range_match(self, evaluator_no_model):
        """Test range value matching"""
        assert evaluator_no_model._is_value_in_range([10, 20], 15) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 10) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 20) is True
        assert evaluator_no_model._is_value_in_range([10, 20], 25) is False

    def test_is_value_in_range_none_values(self, evaluator_no_model):
        """Test None value handling"""
        assert evaluator_no_model._is_value_in_range(None, None) is True
        assert evaluator_no_model._is_value_in_range(None, 10) is False
        assert evaluator_no_model._is_value_in_range(10, None) is False

    def test_is_value_in_range_with_error_thresholds(self, evaluator_no_model):
        """Test numeric comparison using per-range absolute error tolerances"""
        thresholds = {(0, 200): 5, (201, 500): 10}

        # Within tolerance
        assert evaluator_no_model._is_value_in_range(100.0, 104.0, thresholds) is True
        assert evaluator_no_model._is_value_in_range(100.0, 95.5, thresholds) is True

        # Outside tolerance
        assert evaluator_no_model._is_value_in_range(100.0, 110.0, thresholds) is False

        # Exact boundary of tolerance
        assert evaluator_no_model._is_value_in_range(100.0, 105.0, thresholds) is True
        assert evaluator_no_model._is_value_in_range(100.0, 105.1, thresholds) is False

        # Value in second range
        assert evaluator_no_model._is_value_in_range(300.0, 308.0, thresholds) is True
        assert evaluator_no_model._is_value_in_range(300.0, 315.0, thresholds) is False

        # No matching range — falls back to epsilon comparison
        assert evaluator_no_model._is_value_in_range(600.0, 600.0, thresholds) is True
        assert evaluator_no_model._is_value_in_range(600.0, 605.0, thresholds) is False

    def test_is_value_in_range_layered_thresholds(self, evaluator_no_model):
        """Nested/layered ranges: narrowest matching range always wins."""
        thresholds = {
            (-50, 50): 0.5,
            (-150, 150): 8,
            (float("-inf"), float("inf")): 10,
        }
        # Inner range (-50..50) wins for values inside it
        assert evaluator_no_model._is_value_in_range(30.0, 30.3, thresholds) is True   # within 0.5
        assert evaluator_no_model._is_value_in_range(30.0, 31.0, thresholds) is False  # exceeds 0.5

        # Middle range wins for values in 50..150 / -150..-50
        assert evaluator_no_model._is_value_in_range(75.0, 82.0, thresholds) is True   # within 8
        assert evaluator_no_model._is_value_in_range(75.0, 84.0, thresholds) is False  # exceeds 8
        assert evaluator_no_model._is_value_in_range(-100.0, -107.0, thresholds) is True
        assert evaluator_no_model._is_value_in_range(-100.0, -109.0, thresholds) is False

        # Outer (infinite) range wins for values beyond ±150
        assert evaluator_no_model._is_value_in_range(300.0, 309.0, thresholds) is True  # within 10
        assert evaluator_no_model._is_value_in_range(300.0, 311.0, thresholds) is False

        # Reversed tuple notation is equivalent
        thresholds_reversed = {
            (-50, 50): 0.5,
            (150, -150): 8,
            (float("-inf"), float("inf")): 10,
        }
        assert evaluator_no_model._is_value_in_range(75.0, 82.0, thresholds_reversed) is True
        assert evaluator_no_model._is_value_in_range(75.0, 84.0, thresholds_reversed) is False

    def test_evaluate_uses_value_error_thresholds_from_constructor(
        self, temp_json_files
    ):
        """value_error_thresholds set at construction time is used during evaluate()"""
        gt_file, test_file, output_file = temp_json_files

        # Evaluator with loose tolerance: allow ±5 for values in [0, 10]
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False,
            value_error_thresholds={(0, 10): 5},
        )
        assert evaluator.value_error_thresholds == {(0, 10): 5}

        results = evaluator.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
        )
        assert "overall_accuracy" in results

    def test_evaluate_value_error_thresholds_per_call_override(self, temp_json_files):
        """value_error_thresholds passed to evaluate() overrides the instance-level one"""
        gt_file, test_file, output_file = temp_json_files

        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False,
            value_error_thresholds={(0, 10): 1},
        )

        # Pass a different threshold via the evaluate() call
        results = evaluator.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
            value_error_thresholds={(0, 10): 5},
        )
        # After the call the instance-level threshold should have been updated
        assert evaluator.value_error_thresholds == {(0, 10): 5}
        assert "overall_accuracy" in results

    def test_calculate_text_similarity_sequence_matcher(self, evaluator_no_model):
        """Test text similarity with sequence matcher fallback"""
        similarity = evaluator_no_model._calculate_text_similarity(
            "hello world", "hello world"
        )
        assert similarity == 1.0

        similarity = evaluator_no_model._calculate_text_similarity("hello", "world")
        assert 0 <= similarity <= 1

    @patch("torch.nn.functional.cosine_similarity")
    @patch("torch.no_grad")
    def test_calculate_text_similarity_physbert(
        self, mock_no_grad, mock_cosine, evaluator_with_model
    ):
        """Test text similarity with PhysBERT model"""
        mock_cosine.return_value.item.return_value = 0.95

        similarity = evaluator_with_model._calculate_text_similarity(
            "test text", "similar text"
        )
        assert similarity == 0.95

    def test_evaluate_composition_data_exact_match(
        self, evaluator_no_model, sample_composition_data
    ):
        """Test composition data evaluation with exact matches"""
        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_composition_data(
                sample_composition_data, sample_composition_data
            )
        )

        assert score > 0.8  # Should be high score for exact match
        assert details["property_unit"]["match"] is True
        assert details["family"]["match"] is True
        assert details["compositions_property_values"]["match"] is True

    def test_evaluate_composition_data_empty_reference(self, evaluator_no_model):
        """Test composition data evaluation with empty reference"""
        empty_data = {"compositions_property_values": {}}
        test_data = {"compositions_property_values": {"Al": 100}}

        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_composition_data(empty_data, test_data)
        )

        assert abs_metrics["false_positives"] > 0

    def test_evaluate_synthesis_data_exact_match(
        self, evaluator_no_model, sample_synthesis_data
    ):
        """Test synthesis data evaluation with exact matches"""
        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_synthesis_data(
                sample_synthesis_data, sample_synthesis_data
            )
        )

        assert score > 0.8  # Should be high score for exact match
        assert abs_metrics["true_positives"] > 0

    def test_evaluate_synthesis_data_similar_match(self, evaluator_no_model):
        """Test synthesis data evaluation with similar but not exact matches"""
        ref_data = {
            "method": "sol-gel synthesis",
            "precursors": ["aluminum nitrate"],
            "characterization_techniques": ["XRD"],
            "steps": ["Mix precursors"],
        }

        test_data = {
            "method": "sol gel method",  # Similar method
            "precursors": ["aluminum salt"],  # Similar precursor
            "characterization_techniques": ["X-ray diffraction"],  # Similar technique
            "steps": ["Mix starting materials"],  # Similar step
        }

        score, details, abs_metrics, weights = (
            evaluator_no_model._evaluate_synthesis_data(ref_data, test_data)
        )

        # Should get some credit for similarity
        assert 0.2 <= score <= 0.9

    def test_count_all_items(
        self, evaluator_no_model, sample_composition_data, sample_synthesis_data
    ):
        """Test counting all items in a data entry"""
        item = {
            "composition_data": sample_composition_data,
            "synthesis_data": sample_synthesis_data,
        }

        count = evaluator_no_model._count_all_items(item)

        # Should count: property_unit(1) + family(1) + 3 comp_values*2(6) + method(1) + 3 precursors(3) + 3 techniques(3) + steps(2) = 17
        assert count > 10  # At least some reasonable count

    def test_count_all_items_none_data(self, evaluator_no_model):
        """Test counting items with None data"""
        item = {"composition_data": None, "synthesis_data": None}

        count = evaluator_no_model._count_all_items(item)
        assert count == 0

    def test_evaluate_missing_files(self, evaluator_no_model):
        """Test evaluation with missing files"""
        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler):
                evaluator_no_model.evaluate()

    def test_evaluate_missing_model_name(self, evaluator_no_model, temp_json_files):
        """Test evaluation with missing model name"""
        gt_file, test_file, output_file = temp_json_files

        with patch.object(BaseError, "exit_program", lambda self: None):
            with pytest.raises(ValueErrorHandler):
                evaluator_no_model.evaluate(
                    ground_truth_file=gt_file,
                    test_data_file=test_file,
                    extraction_agent_model_name=None,
                )

    def test_evaluate_successful_run(self, evaluator_no_model, temp_json_files):
        """Test successful evaluation run"""
        gt_file, test_file, output_file = temp_json_files

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
            is_synthesis_evaluation=True,
        )

        # Check basic structure of results
        assert "extraction_agent_model_name" in results
        assert "overall_accuracy" in results
        assert "overall_composition_accuracy" in results
        assert "overall_synthesis_accuracy" in results
        assert "absolute_classification_metrics" in results
        assert "normalized_classification_metrics" in results
        assert "item_results" in results

        # Check that results were saved to file
        assert os.path.exists(output_file)

        # Verify file contents
        with open(output_file, "r") as f:
            saved_results = json.load(f)
        assert saved_results["extraction_agent_model_name"] == "test_model"

    def test_evaluate_composition_only(self, evaluator_no_model, temp_json_files):
        """Test evaluation with composition data only"""
        gt_file, test_file, output_file = temp_json_files

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            output_file=output_file,
            is_synthesis_evaluation=False,
        )

        assert "overall_composition_accuracy" in results
        assert (
            results["overall_synthesis_accuracy"] == 0.0
        )  # Should be 0 when not evaluated

    def test_evaluate_custom_weights(self, evaluator_no_model, temp_json_files):
        """Test evaluation with custom weights"""
        gt_file, test_file, output_file = temp_json_files

        custom_weights = {
            "compositions_property_values": 0.4,
            "property_unit": 0.1,
            "family": 0.1,
            "method": 0.1,
            "precursors": 0.1,
            "characterization_techniques": 0.1,
            "steps": 0.1,
        }

        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            weights=custom_weights,
            output_file=output_file,
        )

        assert "overall_accuracy" in results

    def test_evaluate_invalid_weights(self, evaluator_no_model, temp_json_files):
        """Test evaluation with invalid weights (not summing to 1.0)"""
        gt_file, test_file, output_file = temp_json_files

        invalid_weights = {
            "compositions_property_values": 0.5,
            "property_unit": 0.5,  # Total = 1.0, but missing other components
            "family": 0.0,
            "method": 0.0,
            "precursors": 0.0,
            "characterization_techniques": 0.0,
            "steps": 0.0,
        }

        # Should fall back to default weights
        results = evaluator_no_model.evaluate(
            ground_truth_file=gt_file,
            test_data_file=test_file,
            extraction_agent_model_name="test_model",
            weights=invalid_weights,
            output_file=output_file,
        )

        assert "overall_accuracy" in results

    def test_evaluate_missing_dois(self, evaluator_no_model):
        """Test evaluation with DOIs present in only one dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            # Create datasets with different DOIs
            gt_data = {
                "10.1000/only_in_gt": {
                    "composition_data": {
                        "property_unit": "MPa",
                        "family": "ceramics",
                        "compositions_property_values": {"Al2O3": 100},
                    }
                }
            }

            test_data = {
                "10.1000/only_in_test": {
                    "composition_data": {
                        "property_unit": "GPa",
                        "family": "metals",
                        "compositions_property_values": {"Fe": 200},
                    }
                }
            }

            with open(gt_file, "w") as f:
                json.dump(gt_data, f)

            with open(test_file, "w") as f:
                json.dump(test_data, f)

            results = evaluator_no_model.evaluate(
                ground_truth_file=gt_file,
                test_data_file=test_file,
                extraction_agent_model_name="test_model",
                output_file=output_file,
            )

            # Should handle missing DOIs correctly
            assert len(results["item_results"]) == 2  # Both DOIs should be present
            assert results["absolute_classification_metrics"]["false_negatives"] > 0
            assert results["absolute_classification_metrics"]["false_positives"] > 0

    def test_load_models_physbert_failure_fallback_success(self):
        """Test model loading with PhysBERT failure but sentence-transformers success"""
        with (
            patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=ImportError("PhysBERT not available"),
            ),
            patch("sentence_transformers.SentenceTransformer") as mock_st,
        ):

            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)

            assert evaluator.physbert_available is False
            assert evaluator.model_available is True
            mock_st.assert_called_once()

    def test_load_models_all_failures(self):
        """Test model loading with all models failing"""
        with (
            patch(
                "transformers.AutoModel.from_pretrained",
                side_effect=ImportError("PhysBERT not available"),
            ),
            patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=ImportError("SentenceTransformers not available"),
            ),
        ):

            evaluator = MaterialsDataSemanticEvaluator(use_semantic_model=True)

            assert evaluator.physbert_available is False
            assert evaluator.model_available is False

    @pytest.mark.parametrize(
        "threshold_key,threshold_value",
        [
            ("composition_overall_match", 0.75),
            ("synthesis_overall_match", 0.85),
            ("composition_key_match", 0.90),
            ("property_unit_match", 0.95),
        ],
    )
    def test_custom_similarity_thresholds(self, threshold_key, threshold_value):
        """Test custom similarity thresholds"""
        custom_thresholds = {threshold_key: threshold_value}
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False, similarity_thresholds=custom_thresholds
        )

        assert evaluator.similarity_thresholds[threshold_key] == threshold_value

    def test_evaluate_empty_datasets(self, evaluator_no_model):
        """Test evaluation with empty datasets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_file = os.path.join(tmpdir, "ground_truth.json")
            test_file = os.path.join(tmpdir, "test_data.json")
            output_file = os.path.join(tmpdir, "output.json")

            # Create empty datasets
            with open(gt_file, "w") as f:
                json.dump({}, f)

            with open(test_file, "w") as f:
                json.dump({}, f)

            results = evaluator_no_model.evaluate(
                ground_truth_file=gt_file,
                test_data_file=test_file,
                extraction_agent_model_name="test_model",
                output_file=output_file,
            )

            assert results["total_items"] == 0
            assert results["overall_accuracy"] == 0.0
            assert len(results["item_results"]) == 0


BRANCHED_SCHEMA_YAML = """
version: "1.0"
composition_fields:
  - name: compositions_property_values
    type: "dict[str, object]"
    weight: 0.8
    value_fields:
      - name: d33_data
        type: object
        weight: 0.6
        value_fields:
          - {name: value, type: "float|int|None", weight: 0.5}
          - {name: unit, type: str, match_mode: exact}
      - name: grain
        type: object
        weight: 0.4
        value_fields:
          - {name: grain_size, type: "list[float]", similarity_threshold: 0.1, similarity_threshold_mode: absolute}
  - {name: property_unit, type: str, weight: 0.2, match_mode: exact}
synthesis_fields:
  - {name: method, type: str, weight: 0.5}
  - {name: sintering_atmosphere, type: "str|None", weight: 0.5, match_mode: exact}
"""


class TestCustomSchemaEvaluation:
    """Test semantic evaluation with a branched custom schema"""

    @pytest.fixture
    def evaluator(self, tmp_path):
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(BRANCHED_SCHEMA_YAML, encoding="utf-8")
        return MaterialsDataSemanticEvaluator(
            use_semantic_model=False, schema_config_path=schema_path
        )

    @staticmethod
    def composition(d33=190, unit="pC/N", grain=(2.8, 3.1)):
        value = {"d33_data": {"value": d33, "unit": unit}}
        if grain is not None:
            value["grain"] = {"grain_size": list(grain)}
        return value

    def test_default_weights_from_schema(self, evaluator):
        assert evaluator.default_weights == pytest.approx(
            {
                "compositions_property_values": 0.4,
                "property_unit": 0.1,
                "method": 0.25,
                "sintering_atmosphere": 0.25,
            }
        )

    def test_each_leaf_is_one_item(self, evaluator):
        reference = {
            "compositions_property_values": {
                "BaTiO3": self.composition(),
                "KNbO3": self.composition(d33=120),
            },
            "property_unit": "pC/N",
        }
        test = {
            "compositions_property_values": {
                "BaTiO3": self.composition(),
                # Grain sizes outside the absolute 0.1 tolerance
                "KNbO3": self.composition(d33=120, grain=(4.0, 5.0)),
            },
            "property_unit": "pC/N",
        }
        score, details, metrics, _ = evaluator._evaluate_composition_data(
            reference, test
        )
        cpv = details["compositions_property_values"]["details"]
        # 2 keys + 6 matching leaves + property_unit
        assert metrics["true_positives"] == 2 + 5 + 1
        assert metrics["false_positives"] == metrics["false_negatives"] == 1
        assert cpv["leaf_matches"]["KNbO3"]["grain.grain_size"]["match"] is False
        assert cpv["leaf_matches"]["KNbO3"]["d33_data.value"]["match"] is True
        # KNbO3 earns the d33_data share (0.6) of its value: (1 + 0.6) / 2
        assert cpv["pair_match_ratio"] == pytest.approx(0.8)
        assert cpv["value_matches"] == {"BaTiO3": True, "KNbO3": False}

    def test_grain_within_tolerance_matches(self, evaluator):
        reference = {"compositions_property_values": {"BaTiO3": self.composition()}}
        test = {
            "compositions_property_values": {
                "BaTiO3": self.composition(grain=(3.15, 2.85))
            }
        }
        _, details, metrics, _ = evaluator._evaluate_composition_data(reference, test)
        assert details["compositions_property_values"]["match"] is True
        assert metrics["false_positives"] == metrics["false_negatives"] == 0

    def test_missing_composition_counts_key_and_present_leaves(self, evaluator):
        reference = {
            "compositions_property_values": {
                "BaTiO3": self.composition(),
                "KNbO3": self.composition(grain=None),
            }
        }
        test = {"compositions_property_values": {"BaTiO3": self.composition()}}
        _, details, metrics, _ = evaluator._evaluate_composition_data(reference, test)
        # Missing KNbO3: its key plus its 2 present leaves (no grain block)
        assert metrics["false_negatives"] == 3
        assert details["compositions_property_values"]["details"]["missing_keys"] == [
            "KNbO3"
        ]

    def test_custom_synthesis_field(self, evaluator):
        reference = {"method": "sol-gel", "sintering_atmosphere": "air"}
        test = {"method": "sol-gel", "sintering_atmosphere": "oxygen"}
        score, details, metrics, weights = evaluator._evaluate_synthesis_data(
            reference, test
        )
        assert details["sintering_atmosphere"]["match"] is False
        assert weights == pytest.approx({"method": 0.25, "sintering_atmosphere": 0.25})
        assert score == pytest.approx(0.75)

    def test_count_all_items_uses_leaves(self, evaluator):
        item = {
            "composition_data": {
                "compositions_property_values": {"BaTiO3": self.composition()},
                "property_unit": "pC/N",
            },
            "synthesis_data": {"method": "sol-gel"},
        }
        # Key + 3 leaves, property_unit, method
        assert evaluator._count_all_items(item) == 6

    def test_legacy_weights_dict_still_overrides(self, evaluator):
        reference = {"method": "sol-gel", "sintering_atmosphere": "air"}
        test = {"method": "sol-gel", "sintering_atmosphere": "oxygen"}
        weights = {
            "compositions_property_values": 0.5,
            "property_unit": 0.0,
            "method": 0.4,
            "sintering_atmosphere": 0.1,
        }
        score, _, _, _ = evaluator._evaluate_synthesis_data(reference, test, weights)
        assert score == pytest.approx(0.9)

    def test_field_match_threshold_override(self, tmp_path):
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(BRANCHED_SCHEMA_YAML, encoding="utf-8")
        evaluator = MaterialsDataSemanticEvaluator(
            use_semantic_model=False,
            schema_config_path=schema_path,
            similarity_thresholds={"method_match": 0.1},
        )
        method = evaluator.schema_config.synthesis_fields[0]
        assert evaluator._field_threshold(method) == 0.1


class TestNumericTolerance:
    """Test numeric leaf comparison with schema tolerances"""

    @pytest.fixture
    def evaluator(self):
        return MaterialsDataSemanticEvaluator(use_semantic_model=False)

    @pytest.mark.parametrize(
        "mode, threshold, ref, test, expected",
        [
            ("relative", 0.05, 100, 104, True),
            ("relative", 0.05, 100, 106, False),
            ("absolute", 0.1, 2.0, 2.09, True),
            ("absolute", 0.1, 2.0, 2.2, False),
            ("absolute", 0.1, [3.1, 2.8], [2.85, 3.05], True),
            ("absolute", 0.1, [3.1, 2.8], [3.1], False),
        ],
    )
    def test_within_tolerance(self, evaluator, mode, threshold, ref, test, expected):
        from comproscanner.utils.configs.schema_config import SchemaField

        field = SchemaField(
            name="x",
            type="float",
            similarity_threshold=threshold,
            similarity_threshold_mode=mode,
        )
        assert evaluator._is_within_tolerance(field, ref, test) is expected


class TestSwapCorrection:
    """Test re-pairing of similar compositions whose values got interchanged"""

    @pytest.fixture
    def evaluator(self):
        return MaterialsDataSemanticEvaluator(use_semantic_model=False)

    @staticmethod
    def similarities(table):
        """Text similarity stub from a {(ref_key, test_key): similarity} table"""
        return lambda a, b: table.get((a, b), table.get((b, a), 0.0))

    def evaluate(self, evaluator, ref_values, test_values, table):
        with patch.object(
            evaluator, "_calculate_text_similarity", side_effect=self.similarities(table)
        ):
            _, details, metrics, _ = evaluator._evaluate_composition_data(
                {"compositions_property_values": ref_values},
                {"compositions_property_values": test_values},
            )
        return details["compositions_property_values"]["details"], metrics

    def test_two_interchanged_pairs_are_swapped(self, evaluator):
        # Semantic matching alone pairs c1-t1 and c2-t2, whose values are interchanged
        table = {("c1", "t1"): 0.98, ("c1", "t2"): 0.95, ("c2", "t1"): 0.95, ("c2", "t2"): 0.98}
        details, metrics = self.evaluate(
            evaluator, {"c1": 10, "c2": 20}, {"t1": 20, "t2": 10}, table
        )
        assert details["key_matches"]["c1"]["matched_with"] == "t2"
        assert details["key_matches"]["c2"]["matched_with"] == "t1"
        assert {m["match_type"] for m in details["key_matches"].values()} == {"swap_corrected"}
        assert details["value_matches"] == {"c1": True, "c2": True}
        assert metrics == {"true_positives": 4, "false_positives": 0, "false_negatives": 0}

    def test_three_way_rotation_is_corrected(self, evaluator):
        keys = ["c1", "c2", "c3"]
        tests = ["t1", "t2", "t3"]
        table = {(c, t): (0.99 if c[1] == t[1] else 0.93) for c in keys for t in tests}
        details, _ = self.evaluate(
            evaluator,
            {"c1": 10, "c2": 20, "c3": 30},
            {"t1": 30, "t2": 10, "t3": 20},
            table,
        )
        assert {k: m["matched_with"] for k, m in details["key_matches"].items()} == {
            "c1": "t2",
            "c2": "t3",
            "c3": "t1",
        }
        assert all(details["value_matches"].values())

    def test_correction_never_loses_a_key_match(self, evaluator):
        # c1-t1 agree on value, but taking t1 would leave c2 without any partner above 0.9
        table = {("c1", "t1"): 0.95, ("c1", "t2"): 0.97, ("c2", "t1"): 0.96}
        details, _ = self.evaluate(
            evaluator, {"c1": 10, "c2": 20}, {"t1": 10, "t2": 30}, table
        )
        assert details["key_matches"]["c1"]["matched_with"] == "t2"
        assert details["key_matches"]["c2"]["matched_with"] == "t1"
        assert details["missing_keys"] == []

    def test_correct_pairs_are_kept(self, evaluator):
        table = {("c1", "t1"): 0.98, ("c1", "t2"): 0.95, ("c2", "t1"): 0.95, ("c2", "t2"): 0.98}
        details, _ = self.evaluate(
            evaluator, {"c1": 10, "c2": 20}, {"t1": 10, "t2": 20}, table
        )
        assert {m["match_type"] for m in details["key_matches"].values()} == {
            "high_similarity"
        }
        assert details["key_matches"]["c1"]["matched_with"] == "t1"

    def test_results_do_not_depend_on_hash_seed(self, tmp_path):
        ground_truth = {
            "10.1/a": {
                "composition_data": {
                    "compositions_property_values": {
                        f"(Ba0.95Ca0.05)(Ti0.89Sn0.11)O3 + 0.{i}%BiFeO3": 100 + i
                        for i in range(1, 6)
                    }
                }
            }
        }
        test_data = {
            "10.1/a": {
                "composition_data": {
                    "compositions_property_values": {
                        f"(Ba0.95Ca0.05)(Ti0.89Sn0.11)O3+0.{i}% BiFeO3": 100 + (i % 5) + 1
                        for i in range(1, 6)
                    }
                }
            }
        }
        (tmp_path / "gt.json").write_text(json.dumps(ground_truth), encoding="utf-8")
        (tmp_path / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
        runner = (
            "import json, sys\n"
            "from comproscanner.post_processing.evaluation.semantic_evaluator import MaterialsDataSemanticEvaluator\n"
            "r = MaterialsDataSemanticEvaluator(use_semantic_model=False).evaluate("
            "ground_truth_file=sys.argv[1], test_data_file=sys.argv[2], "
            "extraction_agent_model_name='x', output_file=sys.argv[3])\n"
        )
        src_dir = Path(__file__).resolve().parents[3] / "src"
        outputs = []
        for seed in ("1", "2"):
            out = tmp_path / f"out_{seed}.json"
            subprocess.run(
                [sys.executable, "-c", runner, str(tmp_path / "gt.json"), str(tmp_path / "test.json"), str(out)],
                env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(src_dir)},
                check=True,
                capture_output=True,
            )
            outputs.append(out.read_text(encoding="utf-8"))
        assert outputs[0] == outputs[1]


NESTED_DICT_SCHEMA_YAML = """
version: "1.0"
composition_fields:
  - name: compositions_property_values
    type: "dict[str, object]"
    value_fields:
      - name: grain
        type: object
        value_fields:
          - {name: unit, type: str, match_mode: exact}
          - name: measurements
            type: "dict[str, object]"
            value_fields:
              - {name: mean, type: "float|None", similarity_threshold: 0.1}
synthesis_fields:
  - {name: method, type: str}
"""


class TestNestedKeyedDictionaryLeaves:
    """Test leaf-level evaluation of keyed dictionaries nested inside a block"""

    @pytest.fixture
    def evaluator(self, tmp_path):
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(NESTED_DICT_SCHEMA_YAML, encoding="utf-8")
        return MaterialsDataSemanticEvaluator(
            use_semantic_model=False, schema_config_path=schema_path
        )

    @staticmethod
    def evaluate(evaluator, ref_measurements, test_measurements):
        def record(measurements):
            return {
                "compositions_property_values": {
                    "BaTiO3": {"grain": {"unit": "um", "measurements": measurements}}
                }
            }

        _, details, metrics, _ = evaluator._evaluate_composition_data(
            record(ref_measurements), record(test_measurements)
        )
        leaves = details["compositions_property_values"]["details"]["leaf_matches"]["BaTiO3"]
        return leaves, metrics

    def test_each_entry_leaf_is_one_item(self, evaluator):
        leaves, metrics = self.evaluate(
            evaluator,
            {"SEM": {"mean": 18.5}, "EBSD": {"mean": 16.9}},
            {"SEM": {"mean": 21.0}, "EBSD": {"mean": 16.9}},
        )
        assert leaves["grain.measurements.SEM.mean"]["match"] is False
        assert leaves["grain.measurements.EBSD.mean"]["match"] is True
        # Key + unit + EBSD mean
        assert metrics == {"true_positives": 3, "false_positives": 1, "false_negatives": 1}

    def test_entry_only_in_test_counts_as_extra(self, evaluator):
        leaves, metrics = self.evaluate(
            evaluator, {"SEM": {"mean": 18.5}}, {"SEM": {"mean": 18.5}, "TEM": {"mean": 9.0}}
        )
        assert leaves["grain.measurements.TEM.mean"]["match"] is False
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 0
