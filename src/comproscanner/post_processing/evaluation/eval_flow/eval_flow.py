"""
eval_flow.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 20-04-2025
"""

# Standard library imports
from typing import Dict, Optional, Any, List
import json
import os
import time
from pathlib import Path

# Third party imports
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import LLM

from comproscanner.utils.configs import DefaultPaths
from comproscanner.utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    load_schema_config,
)
from comproscanner.utils.error_handler import ValueErrorHandler
from comproscanner.utils.logger import setup_logger
from ..schema_items import count_record_items, count_value_items, leaf_weights
from .crews.composition_evaluation_crew.composition_evaluation_crew import (
    CompositionEvaluationCrew,
)
from .crews.synthesis_evaluation_crew.synthesis_evaluation_crew import (
    SynthesisEvaluationCrew,
)

# Logger Configuration
logger = setup_logger("comproscanner.log", module_name="eval_flow")


class AgentEvaluationState(BaseModel):
    """State model for the AgentMaterialsEvaluationFlow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input files and paths
    ground_truth_file: str = ""
    test_data_file: str = ""
    output_file: str = DefaultPaths.AGENTIC_EVALUATION_RESULT_FILENAME

    # Evaluation parameters
    extraction_agent_model_name: str = "gpt-4o-mini"
    is_synthesis_evaluation: bool = True

    # Data storage
    ground_truth_data: Dict = {}
    test_data: Dict = {}
    existing_results: Dict = {}

    # LLM configuration
    llm: Optional[LLM] = None

    # Weights for evaluation components
    weights: Dict[str, float] = {}

    # Optional per-range absolute error tolerances for numeric property values
    value_error_thresholds: Dict = {}

    # Results storage
    evaluation_details: Dict = {}
    item_results: Dict = {}
    combined_evaluation_results: Dict = {}

    # Progress tracking
    processed_count: int = 0
    total_count: int = 0
    remaining_dois: List[str] = []


class MaterialsDataAgenticEvaluatorFlow(Flow[AgentEvaluationState]):
    """
    CrewAI Flow for evaluating materials data extraction using AI agents that make
    binary (yes/no) decisions about whether items match, rather than using semantic
    similarity or exact matching.

    Args:
        ground_truth_file (str): Path to the ground truth JSON file
        test_data_file (str): Path to the test data JSON file
        output_file (str, optional): Path to save the evaluation results. Default: "agentic_evaluation_result.json"
        extraction_agent_model_name (str, optional): Name of the agent model used in data extraction (default: "gpt-4o-mini")
        is_synthesis_evaluation (bool, optional): Whether to evaluate synthesis data. Default: True
        weights (Dict[str, float], optional): Custom weights for different components. Default: None
        llm (LLM, optional): LLM instance for the agents. Default: None
        schema_config_path (str | Path, optional): Path to the extraction schema YAML file. Default: None (bundled default schema)

    Returns:
        Dict: Detailed evaluation results with binary match decisions and metrics
    """

    def __init__(
        self,
        ground_truth_file: str = None,
        test_data_file: str = None,
        extraction_agent_model_name: str = None,
        output_file: str = DefaultPaths.AGENTIC_EVALUATION_RESULT_FILENAME,
        is_synthesis_evaluation: bool = True,
        weights: Dict[str, float] = None,
        llm: Optional[LLM] = None,
        value_error_thresholds: Dict = None,
        schema_config_path: Optional[str] = None,
    ):

        # Validate required inputs
        if not ground_truth_file:
            raise ValueErrorHandler("Ground truth file path is required")
        if not test_data_file:
            raise ValueErrorHandler("Test data file path is required")
        if not extraction_agent_model_name:
            raise ValueErrorHandler("Used agent model name is required")

        # Validate file existence
        if not os.path.exists(ground_truth_file):
            raise ValueErrorHandler(f"Ground truth file not found: {ground_truth_file}")
        if not os.path.exists(test_data_file):
            raise ValueErrorHandler(f"Test data file not found: {test_data_file}")

        super().__init__()
        self.schema_config = load_schema_config(schema_config_path)
        # Pass schema to crews only when custom (crews default to the bundled one)
        self._crew_schema_kwargs = (
            {}
            if self.schema_config is DEFAULT_SCHEMA_CONFIG
            else {"schema_config": self.schema_config}
        )

        # Set state parameters
        self.state.ground_truth_file = ground_truth_file
        self.state.test_data_file = test_data_file
        self.state.output_file = output_file
        self.state.extraction_agent_model_name = extraction_agent_model_name
        self.state.is_synthesis_evaluation = is_synthesis_evaluation
        self.state.llm = llm

        # Set up weights with defaults from the schema, each section summing to 0.5
        default_weights = {
            field.name: field.effective_weight * 0.5
            for domain in ("composition", "synthesis")
            for field in self.schema_config.fields(domain, final_only=True)
        }
        self.state.weights = default_weights.copy()
        if weights:
            self.state.weights.update(weights)

        # Store optional per-range error tolerances for numeric property values
        self.state.value_error_thresholds = value_error_thresholds or {}

    def _section_fields(self, section):
        """Evaluated schema fields of a section ("composition_data" or "synthesis_data")."""
        return self.schema_config.fields(section.replace("_data", ""), final_only=True)

    def _leaf_fraction(self, field, entries):
        """Weighted fraction of matched leaves in a list of leaf match entries."""
        weights = leaf_weights(field)
        total = sum(weights.get(entry.get("leaf"), 0.0) for entry in entries)
        if total <= 0:
            return 0.0
        matched = sum(
            weights.get(entry.get("leaf"), 0.0)
            for entry in entries
            if entry.get("match_value", 0) == 1
        )
        return matched / total

    def _nested_total_match(self, field, field_data):
        """Matched keys of a nested keyed field, each credited with its matched leaf fraction."""
        matched_keys = {
            key_match.get("reference_key")
            for key_match in field_data.get("key_matches", [])
            if key_match.get("match_value", 0) == 1
        }
        total = 0.0
        for key in matched_keys:
            entries = [
                entry
                for entry in field_data.get("value_matches", [])
                if entry.get("reference_key") == key
            ]
            total += self._leaf_fraction(field, entries)
        return total

    def _calculate_tp_fp_fn(self, details, section):
        """
        Calculate true positives, false positives, and false negatives for a given section.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate metrics for (composition_data or synthesis_data)

        Returns:
            dict: Dictionary with true_positives, false_positives, and false_negatives
        """
        metrics = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}

        if section not in details or not details[section]:
            return metrics

        section_data = details[section]

        for field in self._section_fields(section):
            if field.name not in section_data:
                continue
            field_data = section_data[field.name]
            mode = field.match_mode

            if mode in ("exact", "semantic", "numeric"):
                match_value = field_data.get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += 1
                else:
                    # Count as false positive if test exists, false negative if reference exists
                    if field_data.get("reference") is not None:
                        metrics["false_negatives"] += 1
                    if field_data.get("test") is not None:
                        metrics["false_positives"] += 1

            elif mode == "keyed_values":
                # Process key matches
                if "key_matches" in field_data:
                    for key_match in field_data["key_matches"]:
                        if key_match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if key_match.get("reference_key"):
                                metrics["false_negatives"] += 1
                            if key_match.get("test_key"):
                                metrics["false_positives"] += 1

                # Process value matches (one entry per leaf for nested values)
                if "value_matches" in field_data:
                    for value_match in field_data["value_matches"]:
                        if value_match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if value_match.get("reference_value") is not None:
                                metrics["false_negatives"] += 1
                            if value_match.get("test_value") is not None:
                                metrics["false_positives"] += 1

                # Process missing keys (count as false negatives)
                missing_keys = field_data.get("missing_keys", [])
                metrics["false_negatives"] += len(missing_keys)

                # Process extra keys (count as false positives)
                extra_keys = field_data.get("extra_keys", [])
                metrics["false_positives"] += len(extra_keys)

                # Nested values also count every leaf of a missing or extra key
                if field.value_fields:
                    reference = field_data.get("reference") or {}
                    test = field_data.get("test") or {}
                    metrics["false_negatives"] += sum(
                        count_value_items(field, reference.get(key)) for key in missing_keys
                    )
                    metrics["false_positives"] += sum(
                        count_value_items(field, test.get(key)) for key in extra_keys
                    )

            elif mode == "item_list":
                # Process matches
                if "matches" in field_data:
                    for match in field_data["matches"]:
                        if match.get("match_value", 0) == 1:
                            metrics["true_positives"] += 1
                        else:
                            if match.get("reference_item") is not None:
                                metrics["false_negatives"] += 1
                            if match.get("test_item") is not None:
                                metrics["false_positives"] += 1

                # Count missing items as false negatives
                metrics["false_negatives"] += len(field_data.get("missing_items", []))

                # Count extra items as false positives
                metrics["false_positives"] += len(field_data.get("extra_items", []))

            elif mode == "joined_text":
                text_match = field_data.get("match_value", 0)

                # If text_match is a float between 0 and 1
                if isinstance(text_match, float):
                    if text_match > 0.5:  # Partial match above threshold
                        metrics["true_positives"] += 1
                    else:
                        if field_data.get("reference_steps"):
                            metrics["false_negatives"] += 1
                        if field_data.get("test_steps"):
                            metrics["false_positives"] += 1
                # If text_match is binary (0 or 1)
                elif text_match == 1:
                    metrics["true_positives"] += 1
                else:
                    if field_data.get("reference_steps"):
                        metrics["false_negatives"] += 1
                    if field_data.get("test_steps"):
                        metrics["false_positives"] += 1

            else:
                # Nested object: one item per leaf
                for leaf_match in field_data.get("leaf_matches", []):
                    if leaf_match.get("match_value", 0) == 1:
                        metrics["true_positives"] += 1
                    else:
                        if leaf_match.get("reference_value") is not None:
                            metrics["false_negatives"] += 1
                        if leaf_match.get("test_value") is not None:
                            metrics["false_positives"] += 1

        return metrics

    def _calculate_classification_metrics(self, metrics_dict):
        """
        Calculate precision, recall, and F1 score from classification metrics.

        Args:
            metrics_dict (dict): Dictionary containing true_positives, false_positives, and false_negatives

        Returns:
            None: Updates the metrics_dict in place with precision, recall, and f1_score
        """
        tp = metrics_dict.get("true_positives", 0)
        fp = metrics_dict.get("false_positives", 0)
        fn = metrics_dict.get("false_negatives", 0)

        # Calculate precision
        if tp + fp > 0:
            metrics_dict["precision"] = tp / (tp + fp)
        else:
            metrics_dict["precision"] = 0.0

        # Calculate recall
        if tp + fn > 0:
            metrics_dict["recall"] = tp / (tp + fn)
        else:
            metrics_dict["recall"] = 0.0

        # Calculate F1 score
        precision = metrics_dict["precision"]
        recall = metrics_dict["recall"]

        if precision + recall > 0:
            metrics_dict["f1_score"] = 2 * precision * recall / (precision + recall)
        else:
            metrics_dict["f1_score"] = 0.0

    def _calculate_score(self, details, section, weights):
        """
        Calculate a weighted score for a section based on matches.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate score for
            weights (dict): Weights for different components

        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 1.0

        if section not in details:
            return 0.0

        section_data = details[section]

        for field in self._section_fields(section):
            if field.name not in section_data:
                continue
            field_data = section_data[field.name]
            weight = weights[field.name]
            mode = field.match_mode

            if mode in ("exact", "semantic", "numeric"):
                if field_data["match_value"] == 0:
                    score -= weight

            elif mode in ("keyed_values", "item_list"):
                # Calculate based on total_match and the ground truth total
                total_key = (
                    "total_ground_truth_keys"
                    if mode == "keyed_values"
                    else "total_ground_truth_items"
                )
                total_match = field_data.get("total_match", 0)
                if mode == "keyed_values" and field.value_fields:
                    total_match = self._nested_total_match(field, field_data)
                total_ground_truth = field_data.get(total_key, 0)

                # Calculate match ratio if there are ground truth entries
                if total_ground_truth > 0:
                    match_ratio = total_match / total_ground_truth
                    # Apply penalty proportional to match ratio
                    if match_ratio < 1.0:
                        score -= weight * (1 - match_ratio)
                # If no match info or no ground truth entries, apply full penalty
                elif not total_match and not total_ground_truth:
                    score -= weight

            elif mode == "joined_text":
                text_match = field_data["match_value"]
                # If text_match is a float, use it directly
                if isinstance(text_match, float):
                    if text_match < 1.0:
                        score -= weight * (1 - text_match)
                # If it's binary (0 or 1)
                elif text_match == 0:
                    score -= weight

            else:
                fraction = self._leaf_fraction(field, field_data.get("leaf_matches", []))
                score -= weight * (1 - fraction)

        return max(0.0, score)

    def _load_existing_results(self):
        """
        Load existing results from the output file if it exists.

        Returns:
            Dict: Existing evaluation results or empty dict if file doesn't exist
        """
        if os.path.exists(self.state.output_file):
            try:
                with open(self.state.output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                logger.info(f"Loaded existing results from {self.state.output_file}")
                return existing_results
            except Exception as e:
                logger.warning(f"Error loading existing results: {str(e)}")
                return {}
        return {}

    def _update_results_file(self, combined_results):
        """
        Update the results file with the latest evaluation results.

        Args:
            combined_results (dict): The current combined results to write

        Returns:
            bool: True if successful, False otherwise
        """
        output_path = Path(self.state.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write evaluation results: {str(e)}")
            return False

    def _normalize_weights(self):
        """
        Normalize weights for each category (composition and synthesis) to sum to 0.5.
        This ensures that the weights are balanced while calculating normalized classification scores.
        """
        normalized_weights = {}
        for domain in ("composition", "synthesis"):
            names = [
                field.name
                for field in self.schema_config.fields(domain, final_only=True)
            ]
            # Calculate total weights for the category
            total = sum(self.state.weights[name] for name in names)
            # Normalize weights to sum to 0.5 for the category
            for name in names:
                normalized_weights[name] = (self.state.weights[name] / total) * 0.5

        return normalized_weights

    def _calculate_normalized_metrics(self, details, section, normalized_weights):
        """
        Calculate normalized metrics for a given section based on agent evaluation results.

        Args:
            details (dict): The details section of evaluation results
            section (str): The section to calculate metrics for (composition_data or synthesis_data)
            normalized_weights (dict): Normalized weights for different components

        Returns:
            dict: Dictionary with normalized true_positives, false_positives, and false_negatives
        """
        metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        if section not in details or not details[section]:
            return metrics

        section_data = details[section]

        for field in self._section_fields(section):
            if field.name not in section_data:
                continue
            field_data = section_data[field.name]
            weight = normalized_weights[field.name]
            mode = field.match_mode

            if mode in ("exact", "semantic", "numeric"):
                match_value = field_data.get("match_value", 0)
                if match_value == 1:
                    metrics["true_positives"] += weight
                else:
                    # Split the weight between false positives and false negatives
                    if field_data.get("reference") is not None:
                        metrics["false_negatives"] += weight / 2
                    if field_data.get("test") is not None:
                        metrics["false_positives"] += weight / 2

            elif mode == "keyed_values":
                total_ground_truth_keys = field_data.get("total_ground_truth_keys", 0)

                if total_ground_truth_keys > 0:
                    # Split the component weight between keys and values
                    key_weight = weight / 2
                    value_weight = weight / 2

                    # Calculate per-item weight
                    key_item_weight = key_weight / total_ground_truth_keys
                    value_item_weight = value_weight / total_ground_truth_keys

                    # Nested values share the value weight between leaves
                    leaf_shares = None
                    if field.value_fields:
                        weights = leaf_weights(field)
                        total_leaf_weight = sum(weights.values())
                        leaf_shares = {
                            path: (w / total_leaf_weight if total_leaf_weight else 0.0)
                            for path, w in weights.items()
                        }

                    # Process key matches
                    if "key_matches" in field_data:
                        for key_match in field_data["key_matches"]:
                            if key_match.get("match_value", 0) == 1:
                                metrics["true_positives"] += key_item_weight
                            else:
                                if key_match.get("reference_key"):
                                    metrics["false_negatives"] += key_item_weight / 2
                                if key_match.get("test_key"):
                                    metrics["false_positives"] += key_item_weight / 2

                    # Process value matches
                    if "value_matches" in field_data:
                        for value_match in field_data["value_matches"]:
                            entry_weight = value_item_weight
                            if leaf_shares is not None:
                                entry_weight *= leaf_shares.get(value_match.get("leaf"), 0.0)
                            if value_match.get("match_value", 0) == 1:
                                metrics["true_positives"] += entry_weight
                            else:
                                if value_match.get("reference_value") is not None:
                                    metrics["false_negatives"] += entry_weight / 2
                                if value_match.get("test_value") is not None:
                                    metrics["false_positives"] += entry_weight / 2

                    # Missing keys (just add as false negatives)
                    missing_keys_count = len(field_data.get("missing_keys", []))
                    if missing_keys_count > 0:
                        metrics["false_negatives"] += missing_keys_count * (
                            key_item_weight + value_item_weight
                        )

                    # Extra keys (just add as false positives)
                    extra_keys_count = len(field_data.get("extra_keys", []))
                    if extra_keys_count > 0:
                        metrics["false_positives"] += extra_keys_count * (
                            key_item_weight + value_item_weight
                        )

            elif mode == "item_list":
                total_ground_truth_items = field_data.get("total_ground_truth_items", 0)

                if total_ground_truth_items > 0:
                    # Calculate per-item weight
                    item_weight = weight / total_ground_truth_items

                    # Process matches
                    if "matches" in field_data:
                        for match in field_data["matches"]:
                            if match.get("match_value", 0) == 1:
                                metrics["true_positives"] += item_weight
                            else:
                                if match.get("reference_item") is not None:
                                    metrics["false_negatives"] += item_weight / 2
                                if match.get("test_item") is not None:
                                    metrics["false_positives"] += item_weight / 2

                    # Missing items
                    missing_items_count = len(field_data.get("missing_items", []))
                    if missing_items_count > 0:
                        metrics["false_negatives"] += missing_items_count * item_weight

                    # Extra items
                    extra_items_count = len(field_data.get("extra_items", []))
                    if extra_items_count > 0:
                        metrics["false_positives"] += extra_items_count * item_weight

            elif mode == "joined_text":
                text_match = field_data.get("match_value", 0)

                # If text_match is a float (0.0 to 1.0)
                if isinstance(text_match, float):
                    metrics["true_positives"] += weight * text_match
                    remaining = weight * (1 - text_match)
                    metrics["false_negatives"] += remaining / 2
                    metrics["false_positives"] += remaining / 2
                # If it's binary (0 or 1)
                elif text_match == 1:
                    metrics["true_positives"] += weight
                else:
                    # Split weight between false positives and negatives
                    metrics["false_negatives"] += weight / 2
                    metrics["false_positives"] += weight / 2

            else:
                weights = leaf_weights(field)
                total_leaf_weight = sum(weights.values())
                for leaf_match in field_data.get("leaf_matches", []):
                    share = weights.get(leaf_match.get("leaf"), 0.0)
                    entry_weight = weight * (share / total_leaf_weight if total_leaf_weight else 0.0)
                    if leaf_match.get("match_value", 0) == 1:
                        metrics["true_positives"] += entry_weight
                    else:
                        if leaf_match.get("reference_value") is not None:
                            metrics["false_negatives"] += entry_weight / 2
                        if leaf_match.get("test_value") is not None:
                            metrics["false_positives"] += entry_weight / 2

        return metrics

    def _calculate_key_value_match_ratios(self, details):
        """
        Calculate key-value pair match ratios for keyed fields similar to the semantic evaluator.

        Args:
            details (dict): The evaluation details for a DOI

        Returns:
            dict: Updated details with match ratios
        """
        for section in ("composition_data", "synthesis_data"):
            if section not in details or not details[section]:
                continue
            section_data = details[section]
            for field in self._section_fields(section):
                if field.match_mode != "keyed_values" or field.name not in section_data:
                    continue
                field_data = section_data[field.name]

                # Initialize counters
                key_matches_count = 0
                value_matches_count = 0
                pair_matches_count = 0
                total_ref_keys = field_data.get("total_ground_truth_keys", 0)

                # Count key matches from key_matches section
                if "key_matches" in field_data:
                    for key_match in field_data["key_matches"]:
                        if key_match.get("match_value", 0) == 1:
                            key_matches_count += 1

                # Count value matches from value_matches section
                if "value_matches" in field_data:
                    for value_match in field_data["value_matches"]:
                        if value_match.get("match_value", 0) == 1:
                            value_matches_count += 1

                # Count pair matches from pair_matches section (keys that match AND values that match)
                if "pair_matches" in field_data:
                    for pair_match in field_data["pair_matches"]:
                        if pair_match.get("match_value", 0) == 1:
                            pair_matches_count += 1

                # Calculate ratios if there are reference keys
                if total_ref_keys > 0:
                    key_match_ratio = key_matches_count / total_ref_keys
                    value_match_ratio = value_matches_count / total_ref_keys
                    pair_match_ratio = pair_matches_count / total_ref_keys

                    # Calculate overall match ratio (weighted combination of key and pair match)
                    overall_match_ratio = 0.4 * key_match_ratio + 0.6 * pair_match_ratio
                else:
                    key_match_ratio = 0.0
                    value_match_ratio = 0.0
                    pair_match_ratio = 0.0
                    overall_match_ratio = 0.0

                # Add match ratios to the details
                field_data.update(
                    {
                        "key_match_ratio": key_match_ratio,
                        "value_match_ratio": value_match_ratio,
                        "pair_match_ratio": pair_match_ratio,
                        "overall_match_ratio": overall_match_ratio,
                    }
                )

                # Add a similarity_score for compatibility with the semantic evaluator format
                field_data["similarity_score"] = overall_match_ratio

                # Update the match flag based on overall_match_ratio with a threshold of 0.85
                field_data["match"] = overall_match_ratio > 0.85

        return details

    def _enhance_evaluation_details(self, details):
        """
        Enhance the evaluation details to include match ratios and similarity scores.

        Args:
            details (dict): The original evaluation details

        Returns:
            dict: Enhanced details with additional metrics
        """
        # If details is empty or not a dictionary, return as is
        if not details or not isinstance(details, dict):
            return details

        # Enhance keyed fields with key-value match ratios
        details = self._calculate_key_value_match_ratios(details)

        # Add similarity scores and match flags where needed
        for section in ("composition_data", "synthesis_data"):
            if section not in details or not details[section]:
                continue
            section_data = details[section]
            for field in self._section_fields(section):
                if field.name not in section_data:
                    continue
                field_data = section_data[field.name]
                mode = field.match_mode

                if mode == "semantic":
                    match_value = field_data.get("match_value", 0)
                    field_data["similarity"] = 1.0 if match_value == 1 else 0.0
                    field_data["match"] = match_value == 1

                elif mode == "item_list":
                    total_ground_truth_items = field_data.get("total_ground_truth_items", 0)
                    total_match = field_data.get("total_match", 0)

                    if total_ground_truth_items > 0:
                        field_data["similarity"] = total_match / total_ground_truth_items
                        field_data["match"] = field_data["similarity"] >= 0.8
                    else:
                        # If both reference and test are empty, it's a perfect match
                        if (
                            not field_data.get("reference")
                            or len(field_data.get("reference", [])) == 0
                        ) and (
                            not field_data.get("test")
                            or len(field_data.get("test", [])) == 0
                        ):
                            field_data["similarity"] = 1.0
                            field_data["match"] = True
                        else:
                            field_data["similarity"] = 0.0
                            field_data["match"] = False

                elif mode == "joined_text":
                    # Rename match_value to steps_match for compatibility
                    text_match = field_data.get("match_value", 0)

                    if isinstance(text_match, float):
                        field_data["steps_match"] = "true" if text_match > 0.6 else "false"
                    else:
                        field_data["steps_match"] = "true" if text_match == 1 else "false"

        return details

    def _is_section_empty(self, section_data):
        """
        Helper method to determine if a section is effectively empty

        Args:
            section_data (dict): The section data to check

        Returns:
            bool: True if the section is empty, False otherwise
        """
        if not section_data or not isinstance(section_data, dict):
            return True

        # Check if all values are None, empty dicts, or empty lists
        for key, value in section_data.items():
            if value is not None and value != {} and value != []:
                if isinstance(value, dict) and len(value) > 0:
                    return False
                elif isinstance(value, list) and len(value) > 0:
                    return False
                elif not isinstance(value, (dict, list)) and value:
                    return False

        return True

    def _count_all_items(self, item):
        """Count all individual items in a data entry"""
        return count_record_items(self.schema_config, item)

    @start()
    def load_data(self):
        """Load ground truth and test data files, and check for existing results."""
        logger.info(f"Loading ground truth data from {self.state.ground_truth_file}")
        logger.info(f"Loading test data from {self.state.test_data_file}")

        try:
            # Load ground truth and test data
            with open(self.state.ground_truth_file, "r", encoding="utf-8") as gt_file:
                self.state.ground_truth_data = json.load(gt_file)

            with open(self.state.test_data_file, "r", encoding="utf-8") as test_file:
                self.state.test_data = json.load(test_file)

            # Get all DOIs from ground truth
            all_dois = list(self.state.ground_truth_data.keys())
            self.state.total_count = len(all_dois)

            # Load existing results if available
            self.state.existing_results = self._load_existing_results()

            # Extract already processed DOIs
            already_processed_dois = []
            if (
                self.state.existing_results
                and "item_results" in self.state.existing_results
            ):
                already_processed_dois = list(
                    self.state.existing_results["item_results"].keys()
                )
                self.state.processed_count = len(already_processed_dois)
                logger.info(
                    f"Found {self.state.processed_count} already processed items"
                )

            # Determine remaining DOIs to process
            self.state.remaining_dois = [
                doi for doi in all_dois if doi not in already_processed_dois
            ]

            # Initialize evaluation details and item results from existing data
            if self.state.existing_results:
                if "item_results" in self.state.existing_results:
                    self.state.item_results = self.state.existing_results[
                        "item_results"
                    ]

                # Extract existing evaluation details
                for doi, item_result in self.state.item_results.items():
                    if "details" in item_result:
                        self.state.evaluation_details[doi] = item_result["details"]

            logger.info(
                f"Successfully loaded data: {self.state.total_count} ground truth items, "
                f"{len(self.state.test_data)} test items, "
                f"{len(self.state.remaining_dois)} remaining to process"
            )

            return {
                "ground_truth_count": self.state.total_count,
                "test_data_count": len(self.state.test_data),
                "processed_count": self.state.processed_count,
                "remaining_dois": self.state.remaining_dois,
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueErrorHandler(f"Failed to load data files: {str(e)}")

    @listen(load_data)
    def evaluate_items(self, data_info):
        """Evaluate remaining items in the dataset."""
        # Get all DOIs from the ground truth data
        all_dois = set(self.state.ground_truth_data.keys())

        # Filter to only process DOIs that aren't already in existing results
        if (
            self.state.existing_results
            and "item_results" in self.state.existing_results
        ):
            already_processed_dois = set(
                self.state.existing_results["item_results"].keys()
            )
        else:
            already_processed_dois = set()

        remaining_dois = [doi for doi in all_dois if doi not in already_processed_dois]

        remaining_count = len(remaining_dois)
        logger.info(
            f"Starting evaluation of {remaining_count} remaining items (including missing DOIs)"
        )

        if remaining_count == 0:
            logger.info("No remaining items to process, using existing results")
            return self.state.evaluation_details

        # Get normalized weights for metric calculations
        normalized_weights = self._normalize_weights()

        # Build threshold instructions for the agent task prompt (injected as template var)
        if self.state.value_error_thresholds:
            _threshold_instructions = (
                "IMPORTANT — error tolerance is configured for this evaluation. "
                "For each numeric property value comparison, call the "
                "`get_value_error_threshold` tool with the ground-truth reference value "
                "to retrieve the allowed absolute error. "
                "If the tool returns 'threshold:<N>', mark the value as a match (1) when "
                "|test_value - reference_value| <= N. "
                "If the tool returns 'exact', require exact equality (within 1e-6)."
            )
        else:
            _threshold_instructions = ""

        # Set up the crews
        composition_crew = CompositionEvaluationCrew(
            llm=self.state.llm,
            value_error_thresholds=self.state.value_error_thresholds,
            **self._crew_schema_kwargs,
        ).crew()
        if self.state.is_synthesis_evaluation:
            synthesis_crew = SynthesisEvaluationCrew(
                llm=self.state.llm, **self._crew_schema_kwargs
            ).crew()

        # Initialize the combined results with existing data
        combined_results = {
            "extraction_agent_model_name": self.state.extraction_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": len(all_dois),  # Use all DOIs count
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": self.state.item_results,
        }

        if not self.state.is_synthesis_evaluation:
            combined_results.pop("overall_synthesis_accuracy", None)

        # Process each remaining DOI one at a time
        for doi_index, doi in enumerate(
            tqdm(remaining_dois, desc="Processing DOIs", unit="item")
        ):
            logger.info(
                f"Evaluating item {doi_index+1}/{remaining_count} with DOI: {doi}"
            )

            # Get ground truth and test data for this DOI
            ground_truth_item = self.state.ground_truth_data.get(doi, {})
            test_item = self.state.test_data.get(doi, {})

            # Initialize results for this DOI
            item_result = {
                "overall_match": False,
                "field_scores": {},
                "overall_score": 0.0,
                "absolute_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "normalized_classification_metrics": {
                    "true_positives": 0.0,
                    "false_positives": 0.0,
                    "false_negatives": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": {},
            }

            # Case 1: DOI exists in both datasets
            if ground_truth_item and test_item:
                doi_details = {}
                has_valid_data = False

                # Check if composition data exists in both ground truth and test items
                has_composition_data = (
                    "composition_data" in ground_truth_item
                    and ground_truth_item["composition_data"]
                    and "composition_data" in test_item
                    and test_item["composition_data"]
                )

                # Evaluate composition data if it exists
                if has_composition_data:
                    composition_result = composition_crew.kickoff(
                        inputs={
                            "ground_truth_item": json.dumps(
                                ground_truth_item.get("composition_data", {})
                            ),
                            "test_item": json.dumps(
                                test_item.get("composition_data", {})
                            ),
                            "value_error_threshold_instructions": _threshold_instructions,
                        }
                    )

                    # Store composition evaluation details
                    try:
                        if isinstance(composition_result.raw, str):
                            composition_details = json.loads(composition_result.raw)
                        else:
                            composition_details = composition_result.raw

                        # Check if composition_data is in the result
                        if "composition_data" in composition_details:
                            doi_details["composition_data"] = composition_details[
                                "composition_data"
                            ]
                            if not self._is_section_empty(
                                doi_details["composition_data"]
                            ):
                                has_valid_data = True
                        else:
                            doi_details["composition_data"] = composition_details
                            if not self._is_section_empty(
                                doi_details["composition_data"]
                            ):
                                has_valid_data = True
                    except Exception as e:
                        logger.error(
                            f"Error processing composition results for DOI {doi}: {str(e)}"
                        )
                        doi_details["composition_data"] = {}

                # Check if synthesis data exists in both ground truth and test items
                has_synthesis_data = (
                    self.state.is_synthesis_evaluation
                    and "synthesis_data" in ground_truth_item
                    and ground_truth_item["synthesis_data"]
                    and "synthesis_data" in test_item
                    and test_item["synthesis_data"]
                )

                # Evaluate synthesis data if it exists
                if has_synthesis_data:
                    synthesis_result = synthesis_crew.kickoff(
                        inputs={
                            "ground_truth_item": json.dumps(
                                ground_truth_item.get("synthesis_data", {})
                            ),
                            "test_item": json.dumps(
                                test_item.get("synthesis_data", {})
                            ),
                        }
                    )

                    # Store synthesis evaluation details
                    try:
                        if isinstance(synthesis_result.raw, str):
                            synthesis_details = json.loads(synthesis_result.raw)
                        else:
                            synthesis_details = synthesis_result.raw

                        # Check if synthesis_data is in the result
                        if "synthesis_data" in synthesis_details:
                            doi_details["synthesis_data"] = synthesis_details[
                                "synthesis_data"
                            ]
                            if not self._is_section_empty(
                                doi_details["synthesis_data"]
                            ):
                                has_valid_data = True
                        else:
                            doi_details["synthesis_data"] = synthesis_details
                            if not self._is_section_empty(
                                doi_details["synthesis_data"]
                            ):
                                has_valid_data = True
                    except Exception as e:
                        logger.error(
                            f"Error processing synthesis results for DOI {doi}: {str(e)}"
                        )
                        doi_details["synthesis_data"] = {}

                # Process valid data if found
                if has_valid_data:
                    # Store evaluation details for this DOI
                    self.state.evaluation_details[doi] = doi_details

                    # Enhance details with match ratios and similarity scores
                    enhanced_details = self._enhance_evaluation_details(doi_details)
                    item_result["details"] = enhanced_details

                    # Calculate composition score
                    if (
                        "composition_data" in enhanced_details
                        and not self._is_section_empty(
                            enhanced_details["composition_data"]
                        )
                    ):
                        composition_score = self._calculate_score(
                            enhanced_details, "composition_data", self.state.weights
                        )
                        item_result["field_scores"][
                            "composition_data"
                        ] = composition_score

                        # Calculate composition metrics
                        comp_metrics = self._calculate_tp_fp_fn(
                            enhanced_details, "composition_data"
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["absolute_classification_metrics"][
                                metric
                            ] += comp_metrics[metric]

                        # Calculate normalized metrics for composition
                        comp_normalized_metrics = self._calculate_normalized_metrics(
                            enhanced_details, "composition_data", normalized_weights
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["normalized_classification_metrics"][
                                metric
                            ] += comp_normalized_metrics[metric]

                    # Calculate synthesis score if applicable
                    if (
                        self.state.is_synthesis_evaluation
                        and "synthesis_data" in enhanced_details
                        and not self._is_section_empty(
                            enhanced_details["synthesis_data"]
                        )
                    ):
                        synthesis_score = self._calculate_score(
                            enhanced_details, "synthesis_data", self.state.weights
                        )
                        item_result["field_scores"]["synthesis_data"] = synthesis_score

                        # Calculate synthesis metrics
                        synth_metrics = self._calculate_tp_fp_fn(
                            enhanced_details, "synthesis_data"
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["absolute_classification_metrics"][
                                metric
                            ] += synth_metrics[metric]

                        # Calculate normalized metrics for synthesis
                        synth_normalized_metrics = self._calculate_normalized_metrics(
                            enhanced_details, "synthesis_data", normalized_weights
                        )
                        for metric in [
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ]:
                            item_result["normalized_classification_metrics"][
                                metric
                            ] += synth_normalized_metrics[metric]

                    # Calculate overall score for the item
                    if (
                        self.state.is_synthesis_evaluation
                        and "composition_data" in item_result["field_scores"]
                        and "synthesis_data" in item_result["field_scores"]
                    ):
                        item_result["overall_score"] = 0.5 * item_result[
                            "field_scores"
                        ].get("composition_data", 0.0) + 0.5 * item_result[
                            "field_scores"
                        ].get(
                            "synthesis_data", 0.0
                        )
                    elif "composition_data" in item_result["field_scores"]:
                        item_result["overall_score"] = item_result["field_scores"].get(
                            "composition_data", 0.0
                        )
                    elif "synthesis_data" in item_result["field_scores"]:
                        item_result["overall_score"] = item_result["field_scores"].get(
                            "synthesis_data", 0.0
                        )

                    # Determine if overall match (threshold: 0.8)
                    item_result["overall_match"] = item_result["overall_score"] > 0.8
                else:
                    # No valid data found - treat as no match
                    item_result["overall_score"] = 0.0
                    item_result["overall_match"] = False
                    logger.warning(f"DOI {doi}: No valid evaluation data found")

            # Case 2: DOI exists in ground truth but not in test (missing)
            elif ground_truth_item and not test_item:
                item_count = self._count_all_items(ground_truth_item)

                # All items are false negatives
                item_result["absolute_classification_metrics"][
                    "false_negatives"
                ] = item_count
                item_result["normalized_classification_metrics"][
                    "false_negatives"
                ] = 1.0  # Full DOI weight
                item_result["overall_score"] = 0.0
                item_result["overall_match"] = False

                # Set field scores to zero
                item_result["field_scores"]["composition_data"] = 0.0
                if self.state.is_synthesis_evaluation:
                    item_result["field_scores"]["synthesis_data"] = 0.0

                logger.info(
                    f"DOI {doi}: Missing from test data (false negatives: {item_count})"
                )

            # Calculate precision, recall, F1 for absolute metrics
            self._calculate_classification_metrics(
                item_result["absolute_classification_metrics"]
            )

            # Calculate precision, recall, F1 for normalized metrics
            self._calculate_classification_metrics(
                item_result["normalized_classification_metrics"]
            )

            # Store item result
            self.state.item_results[doi] = item_result
            combined_results["item_results"][doi] = item_result

            # Update combined metrics after each item
            self._update_combined_metrics(combined_results)

            # Write updated results to file after each item
            self._update_results_file(combined_results)

            logger.info(
                f"Item {doi} processed. "
                f"Score: {item_result['overall_score']:.4f}, "
                f"Absolute F1: {item_result['absolute_classification_metrics']['f1_score']:.4f}, "
                f"Normalized F1: {item_result['normalized_classification_metrics']['f1_score']:.4f}"
            )

        logger.info(f"Completed evaluation of {remaining_count} items")
        time.sleep(5)
        return self.state.evaluation_details

    def _update_combined_metrics(self, combined_results):
        """
        Update the combined metrics based on current item results.

        Args:
            combined_results (dict): The combined results to update
        """
        # Reset accumulator metrics
        total_absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        total_normalized_metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        # Reset scores
        total_composition_score = 0.0
        total_synthesis_score = 0.0
        composition_count = 0
        synthesis_count = 0

        # Sum up all metrics
        for doi, item in combined_results["item_results"].items():
            # Update absolute metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                total_absolute_metrics[metric] += item[
                    "absolute_classification_metrics"
                ].get(metric, 0)

                # Also update normalized metrics
                total_normalized_metrics[metric] += item[
                    "normalized_classification_metrics"
                ].get(metric, 0.0)

            # Update scores
            if "composition_data" in item["field_scores"]:
                total_composition_score += item["field_scores"]["composition_data"]
                composition_count += 1

            if "synthesis_data" in item["field_scores"]:
                total_synthesis_score += item["field_scores"]["synthesis_data"]
                synthesis_count += 1

        # Update combined absolute metrics
        for metric in ["true_positives", "false_positives", "false_negatives"]:
            combined_results["absolute_classification_metrics"][metric] = (
                total_absolute_metrics[metric]
            )

            # Also update normalized metrics
            combined_results["normalized_classification_metrics"][metric] = (
                total_normalized_metrics[metric]
            )

        # Calculate precision, recall, F1 for absolute metrics
        self._calculate_classification_metrics(
            combined_results["absolute_classification_metrics"]
        )

        # Calculate precision, recall, F1 for normalized metrics
        self._calculate_classification_metrics(
            combined_results["normalized_classification_metrics"]
        )

        # Calculate overall accuracies
        total_items = len(combined_results["item_results"])

        # Calculate composition accuracy if composition data is available
        if composition_count > 0:
            combined_results["overall_composition_accuracy"] = (
                total_composition_score / composition_count
            )
        else:
            combined_results["overall_composition_accuracy"] = 0.0

        # Calculate synthesis accuracy if synthesis data is available and it's requested
        if self.state.is_synthesis_evaluation:
            if synthesis_count > 0:
                combined_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / synthesis_count
                )
            else:
                combined_results["overall_synthesis_accuracy"] = 0.0

            # Calculate overall accuracy as average of composition and synthesis
            if composition_count > 0 and synthesis_count > 0:
                combined_results["overall_accuracy"] = (
                    combined_results["overall_composition_accuracy"]
                    + combined_results["overall_synthesis_accuracy"]
                ) / 2
            elif composition_count > 0:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_composition_accuracy"
                ]
            elif synthesis_count > 0:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_synthesis_accuracy"
                ]
            else:
                combined_results["overall_accuracy"] = 0.0
        else:
            # If synthesis evaluation is not enabled, overall is just composition
            combined_results["overall_accuracy"] = combined_results[
                "overall_composition_accuracy"
            ]

    @listen(evaluate_items)
    def finalize_evaluation(self, evaluation_details):
        """
        Finalize the evaluation process and return the final results.

        Args:
            evaluation_details (dict): Combined evaluation details by DOI

        Returns:
            dict: Final evaluation results with metrics and scores
        """
        logger.info("Finalizing agent-based material data evaluation")

        # Get normalized weights
        normalized_weights = self._normalize_weights()

        # Calculate final combined metrics
        combined_results = {
            "extraction_agent_model_name": self.state.extraction_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": len(self.state.item_results),
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0.0,
                "false_positives": 0.0,
                "false_negatives": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": {},
        }

        if not self.state.is_synthesis_evaluation:
            combined_results.pop("overall_synthesis_accuracy", None)

        # Reset accumulator metrics
        total_absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        total_normalized_metrics = {
            "true_positives": 0.0,
            "false_positives": 0.0,
            "false_negatives": 0.0,
        }

        # Reset scores
        total_composition_score = 0.0
        total_synthesis_score = 0.0

        # Process each item's results
        for doi, item in self.state.item_results.items():
            # Enhance the evaluation details with match ratios and similarity scores
            item_details = self._enhance_evaluation_details(item.get("details", {}))

            # Create enhanced item result
            enhanced_item = {
                "overall_match": item.get("overall_match", False),
                "field_scores": item.get("field_scores", {}),
                "overall_score": item.get("overall_score", 0.0),
                "absolute_classification_metrics": item.get(
                    "absolute_classification_metrics",
                    {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                    },
                ),
                "normalized_classification_metrics": {
                    "true_positives": 0.0,
                    "false_positives": 0.0,
                    "false_negatives": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": item_details,
            }

            # Calculate normalized metrics for this item
            comp_normalized_metrics = self._calculate_normalized_metrics(
                item_details, "composition_data", normalized_weights
            )

            # Add composition normalized metrics to item metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                enhanced_item["normalized_classification_metrics"][
                    metric
                ] += comp_normalized_metrics[metric]

            # Add synthesis normalized metrics if applicable
            if self.state.is_synthesis_evaluation:
                synth_normalized_metrics = self._calculate_normalized_metrics(
                    item_details, "synthesis_data", normalized_weights
                )

                for metric in ["true_positives", "false_positives", "false_negatives"]:
                    enhanced_item["normalized_classification_metrics"][
                        metric
                    ] += synth_normalized_metrics[metric]

            # Calculate precision, recall, F1 for normalized metrics
            self._calculate_classification_metrics(
                enhanced_item["normalized_classification_metrics"]
            )

            # Update the combined results item
            combined_results["item_results"][doi] = enhanced_item

            # Update absolute metrics totals
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                total_absolute_metrics[metric] += enhanced_item[
                    "absolute_classification_metrics"
                ].get(metric, 0)
                total_normalized_metrics[metric] += enhanced_item[
                    "normalized_classification_metrics"
                ].get(metric, 0)

            # Update score totals
            total_composition_score += enhanced_item["field_scores"].get(
                "composition_data", 0.0
            )
            if self.state.is_synthesis_evaluation:
                total_synthesis_score += enhanced_item["field_scores"].get(
                    "synthesis_data", 0.0
                )

        # Update combined metrics
        for metric in ["true_positives", "false_positives", "false_negatives"]:
            combined_results["absolute_classification_metrics"][metric] = (
                total_absolute_metrics[metric]
            )
            combined_results["normalized_classification_metrics"][metric] = (
                total_normalized_metrics[metric]
            )

        # Calculate precision, recall, F1 for absolute metrics
        self._calculate_classification_metrics(
            combined_results["absolute_classification_metrics"]
        )

        # Calculate precision, recall, F1 for normalized metrics
        self._calculate_classification_metrics(
            combined_results["normalized_classification_metrics"]
        )

        # Calculate overall accuracies
        total_items = len(combined_results["item_results"])
        if total_items > 0:
            combined_results["overall_composition_accuracy"] = (
                total_composition_score / total_items
            )

            if self.state.is_synthesis_evaluation:
                combined_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / total_items
                )
                combined_results["overall_accuracy"] = (
                    0.5 * combined_results["overall_composition_accuracy"]
                    + 0.5 * combined_results["overall_synthesis_accuracy"]
                )
            else:
                combined_results["overall_accuracy"] = combined_results[
                    "overall_composition_accuracy"
                ]

        # Write final results to file
        self._update_results_file(combined_results)

        # Log summary statistics
        logger.info(f"Overall accuracy: {combined_results['overall_accuracy']:.4f}")
        logger.info(
            f"Composition accuracy: {combined_results['overall_composition_accuracy']:.4f}"
        )

        if (
            self.state.is_synthesis_evaluation
            and "overall_synthesis_accuracy" in combined_results
        ):
            logger.info(
                f"Synthesis accuracy: {combined_results['overall_synthesis_accuracy']:.4f}"
            )

        # Log F1 scores
        abs_f1 = combined_results["absolute_classification_metrics"]["f1_score"]
        norm_f1 = combined_results["normalized_classification_metrics"]["f1_score"]
        logger.info(f"Absolute F1 score: {abs_f1:.4f}")
        logger.info(f"Normalized F1 score: {norm_f1:.4f}")

        logger.info(f"Evaluation complete. Results saved to {self.state.output_file}")

        # Store in state and return
        self.state.combined_evaluation_results = combined_results
        return combined_results
