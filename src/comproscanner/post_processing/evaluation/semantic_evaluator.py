"""
semantic_evaluator.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 16-04-2025
"""

# Standard library imports
import json
import string
from difflib import SequenceMatcher
from pathlib import Path

# Custom imports
from ...utils.configs import CustomDictionary, DefaultPaths
from ...utils.configs.schema_config import load_schema_config
from .schema_items import count_record_items, count_value_items, leaf_values
from ...utils.error_handler import ValueErrorHandler

class MaterialsDataSemanticEvaluator:
    """
    A class to evaluate materials science data against reference data using semantic matching and exact matching (for specific items).

    Args:
        use_semantic_model (bool, optional): Whether to attempt to use semantic models for similarity (default: True)
        primary_model_name (str, optional): Name of the primary model to use (default: "thellert/physbert_cased")
        fallback_model_name (str, optional): Name of the fallback sentence transformer model (default: "all-mpnet-base-v2")
        similarity_thresholds (dict, optional): Custom thresholds for similarity scoring
        schema_config_path (str | Path, optional): Path to the extraction schema YAML file (default: bundled default schema)
    """

    def __init__(
        self,
        use_semantic_model=True,
        primary_model_name="thellert/physbert_cased",
        fallback_model_name="all-mpnet-base-v2",
        similarity_thresholds=None,
        value_error_thresholds=None,
        schema_config_path=None,
    ):
        """
        Initialize the evaluator with optional semantic models.

        Args:
            use_semantic_model (bool, optional): Whether to attempt to use semantic models
            primary_model_name (str, optional): Name of the primary model to use
            fallback_model_name (str, optional): Name of the fallback sentence transformer model
            similarity_thresholds (dict, optional): Custom thresholds for similarity scoring
            value_error_thresholds (dict, optional): Mapping of ground-truth value ranges to
                absolute error tolerances. Keys must be 2-element tuples (min, max); values
                are the allowed absolute difference between the ground-truth and extracted
                numeric property value.  Use float('inf') / float('-inf') for open-ended
                ranges.  Example::

                    {
                        (-200, 200): 5,
                        (201, 500): 8,
                        (-500, -201): 8,
                        (501, float('inf')): 10,
                        (float('-inf'), -501): 10,
                    }

                When a ground-truth value falls inside one of the ranges the extracted
                value is accepted if ``|extracted - ground_truth| <= threshold``.
                If no range matches, exact comparison (epsilon 1e-6) is used.
            schema_config_path (str | Path, optional): Path to the extraction schema YAML file. Defaults to None (bundled default schema).
        """
        self.schema_config = load_schema_config(schema_config_path)
        self.use_semantic_model = use_semantic_model
        self.primary_model_name = primary_model_name
        self.fallback_model_name = fallback_model_name
        self.physbert_available = False
        self.model_available = False
        self.value_error_thresholds = value_error_thresholds or {}

        # Load models if requested
        if self.use_semantic_model:
            self._load_models()

        # Setting similarity thresholds for different components (default or custom)
        self.default_similarity_thresholds = {
            "composition_overall_match": 0.80,
            "synthesis_overall_match": 0.80,
            "composition_key_match": 0.80,
            "property_unit_match": 0.80,
            "family_match": 0.80,
            "method_match": 0.80,
            "precursors_match": 0.80,
            "characterization_match": 0.80,
            "steps_match": 0.80,
        }
        self.similarity_thresholds = self.default_similarity_thresholds.copy()
        if similarity_thresholds:
            self.similarity_thresholds.update(similarity_thresholds)
        # Only user-supplied '<field>_match' keys override schema thresholds
        self.user_similarity_thresholds = dict(similarity_thresholds or {})

        # Default weights from the schema, each section summing to 0.5
        self.default_weights = {
            field.name: field.effective_weight * 0.5
            for domain in ("composition", "synthesis")
            for field in self.schema_config.fields(domain, final_only=True)
        }

    def _load_models(self):
        """
        Attempt to load the semantic models in order of preference.
        First tries to load PhysBERT, then falls back to sentence-transformers.
        """
        # Try to load PhysBERT first
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self.physbert_tokenizer = AutoTokenizer.from_pretrained(
                self.primary_model_name
            )
            self.physbert_model = AutoModel.from_pretrained(self.primary_model_name)
            self.torch = torch
            self.physbert_available = True
        except Exception as e:
            print(f"PhysBERT model not available: {e}, trying sentence-transformers")
            self.physbert_available = False

            # Try sentence-transformers as backup
            if not self.physbert_available:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    self.model = SentenceTransformer(self.fallback_model_name)
                    self.torch = torch
                    self.model_available = True
                except Exception as e:
                    print(
                        f"Sentence-transformer model not available: {e}, falling back to sequence matching"
                    )
                    self.model_available = False

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts.
        Uses PhysBERT if available, falls back to transformer model,
        otherwise uses sequence matching.
        """
        if self.physbert_available:
            # Use PhysBERT model for domain-specific similarity calculation
            inputs1 = self.physbert_tokenizer(
                text1, return_tensors="pt", padding=True, truncation=True
            )
            inputs2 = self.physbert_tokenizer(
                text2, return_tensors="pt", padding=True, truncation=True
            )

            with self.torch.no_grad():
                outputs1 = self.physbert_model(**inputs1)
                outputs2 = self.physbert_model(**inputs2)

            # Use CLS token embeddings for sentence representation
            embedding1 = outputs1.last_hidden_state[:, 0, :]
            embedding2 = outputs2.last_hidden_state[:, 0, :]

            # Normalize embeddings
            embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
            embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)

            # Calculate cosine similarity
            similarity = self.torch.nn.functional.cosine_similarity(
                embedding1, embedding2, dim=1
            ).item()

            return similarity

        elif self.model_available:
            # Fall back to sentence-transformer model
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = self.torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1
            ).item()

            return similarity
        else:
            # Last resort: sequence matcher fallback
            text1_proc = self._simple_preprocess(text1)
            text2_proc = self._simple_preprocess(text2)
            return SequenceMatcher(None, text1_proc, text2_proc).ratio()

    def _simple_preprocess(self, text):
        """
        Simple text preprocessing without external dependencies

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        common_stops = CustomDictionary.get_common_stops

        words = text.split()
        filtered_words = [word for word in words if word not in common_stops]

        return " ".join(filtered_words)

    def _get_error_threshold(self, ref_val, error_thresholds):
        """
        Return the absolute error threshold for *ref_val* from *error_thresholds*.

        Args:
            ref_val (float): Ground-truth numeric value.
            error_thresholds (dict): Mapping of ``(min, max)`` tuples to absolute
                thresholds.  Infinity values are supported.  Ranges are interpreted
                as layers: the narrowest range that contains ``ref_val`` wins, so
                ``(-150, 150): 8`` matches values in (-150, -50) and (50, 150) when
                ``(-50, 50): 0.5`` is also present.  Tuple order does not matter —
                ``(-150, 150)`` and ``(150, -150)`` are equivalent.

        Returns:
            float or None: Threshold if a range contains *ref_val*, else ``None``.
        """
        sorted_ranges = sorted(
            error_thresholds.items(),
            key=lambda item: max(item[0]) - min(item[0]),
        )
        for range_key, threshold in sorted_ranges:
            lo = min(range_key)
            hi = max(range_key)
            if lo <= ref_val <= hi:
                return threshold
        return None

    def _is_value_in_range(self, ref_val, test_val, error_thresholds=None):
        """
        Check if test_val is within the range specified by ref_val, optionally
        using a per-range absolute error tolerance.

        Args:
            ref_val: Reference value, which can be a number or a list [min, max]
            test_val: Test value to check against the reference
            error_thresholds (dict, optional): Mapping of ``(min, max)`` tuples to
                absolute error tolerances (see class docstring).  When provided and a
                matching range is found the comparison becomes
                ``|ref_val - test_val| <= threshold`` instead of an exact check.

        Returns:
            bool: True if the test value matches or falls within the reference range,
                False otherwise.
        """
        # Handle case where either value is None
        if ref_val is None or test_val is None:
            return ref_val == test_val

        # Case 1: ref_val is a list defining a range [min, max]
        if isinstance(ref_val, list) and len(ref_val) == 2:
            try:
                if not isinstance(test_val, (int, float)):
                    test_val = float(test_val)
                return float(ref_val[0]) <= test_val <= float(ref_val[1])
            except (ValueError, TypeError):
                # If conversion fails, do exact comparison
                return ref_val == test_val

        # Case 2: ref_val is not a range, do regular comparison
        try:
            if isinstance(test_val, (int, float)) and isinstance(ref_val, (int, float)):
                ref_num = float(ref_val)
                test_num = float(test_val)
                # Use error threshold dict if provided and a range matches
                if error_thresholds:
                    threshold = self._get_error_threshold(ref_num, error_thresholds)
                    if threshold is not None:
                        return abs(ref_num - test_num) <= threshold
                # Default: floating-point epsilon comparison
                return abs(ref_num - test_num) < 1e-6
            else:
                return ref_val == test_val
        except (ValueError, TypeError):
            # If conversion fails, fall back to exact comparison
            return ref_val == test_val

    def _resolve_weights(self, weights):
        """Return weights to use, falling back to defaults if missing or not summing to 1.0."""
        if weights is None:
            return self.default_weights
        # Check if the sum of weights is approximately 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:  # Allow tiny floating point differences
            print(
                f"Warning: Provided weights sum to {weight_sum}, not 1.0. Using default weights."
            )
            return self.default_weights
        return {**self.default_weights, **weights}

    def _field_threshold(self, field):
        """Return a field's similarity threshold, honouring '<field>_match' overrides."""
        return self.user_similarity_thresholds.get(
            f"{field.name}_match", field.similarity_threshold
        )

    def _evaluate_domain(self, domain, reference, test, weights=None):
        """
        Evaluate one data section field by field according to the extraction schema.

        Args:
            domain (str): "composition" or "synthesis"
            reference (dict): Reference section data
            test (dict): Test section data
            weights (dict, optional): Weights for different components

        Returns:
            tuple: (score, details, absolute_metrics, component_weights)
        """
        weights = self._resolve_weights(weights)
        fields = self.schema_config.fields(domain, final_only=True)

        # Initialize component weights to track relative importance for normalization
        component_weights = {f.name: weights[f.name] for f in fields}

        # Normalize component weights to sum to 0.5 (section portion)
        total_weight = sum(component_weights.values())
        if total_weight > 0:
            for key in component_weights:
                component_weights[key] = component_weights[key] / total_weight * 0.5

        score = 1.0
        details = {}
        absolute_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        evaluators = {
            "exact": self._evaluate_scalar_field,
            "numeric": self._evaluate_scalar_field,
            "semantic": self._evaluate_semantic_field,
            "item_list": self._evaluate_item_list_field,
            "joined_text": self._evaluate_joined_text_field,
            "keyed_values": self._evaluate_keyed_values_field,
            None: self._evaluate_object_field,
        }
        for field in fields:
            penalty, field_details, field_metrics = evaluators[field.match_mode](
                field,
                reference.get(field.name, self._empty_value(field)),
                test.get(field.name, self._empty_value(field)),
                component_weights[field.name],
            )
            score -= penalty
            details[field.name] = field_details
            for metric in absolute_metrics:
                absolute_metrics[metric] += field_metrics[metric]

        return max(0, score), details, absolute_metrics, component_weights

    def _evaluate_composition_data(self, reference_comp, test_comp, weights=None):
        """
        Evaluate composition data section with semantic matching for keys and exact matching for values.

        Args:
            reference_comp (dict): Reference composition data
            test_comp (dict): Test composition data
            weights (dict, optional): Weights for different components

        Returns:
            tuple: (score, details, absolute_metrics, component_weights)
        """
        return self._evaluate_domain("composition", reference_comp, test_comp, weights)

    def _evaluate_synthesis_data(self, reference_synth, test_synth, weights=None):
        """
        Evaluate synthesis data section with semantic understanding.

        Args:
            reference_synth (dict): Reference synthesis data
            test_synth (dict): Test synthesis data
            weights (dict, optional): Weights for different components

        Returns:
            tuple: (score, details, absolute_metrics, component_weights)
        """
        return self._evaluate_domain("synthesis", reference_synth, test_synth, weights)

    @staticmethod
    def _new_metrics():
        return {"true_positives": 0, "false_positives": 0, "false_negatives": 0}

    @staticmethod
    def _empty_value(field):
        """Value used when a field is absent from a record."""
        if field.match_mode in ("item_list", "joined_text"):
            return []
        if field.match_mode == "keyed_values":
            return {}
        return None

    def _values_match(self, field, ref_val, test_val):
        """Check whether two values of a field match under its match mode."""
        if field.match_mode == "numeric":
            if field.similarity_threshold is None:
                return self._is_value_in_range(
                    ref_val, test_val, self.value_error_thresholds
                )
            return self._is_within_tolerance(field, ref_val, test_val)
        if field.match_mode == "semantic":
            if ref_val is None or test_val is None:
                return ref_val == test_val
            return (
                self._calculate_text_similarity(ref_val, test_val)
                >= self._field_threshold(field)
            )
        if field.match_mode == "item_list":
            return self._compare_item_lists(field, ref_val or [], test_val or [])[
                "match"
            ]
        if field.match_mode == "joined_text":
            if not ref_val or not test_val:
                return not ref_val and not test_val
            return self._calculate_text_similarity(
                " ".join(ref_val), " ".join(test_val)
            ) >= self._field_threshold(field)
        return ref_val == test_val

    def _is_within_tolerance(self, field, ref_val, test_val):
        """Compare numbers (or lists of numbers) using the field's similarity threshold."""
        if ref_val is None or test_val is None:
            return ref_val == test_val
        if isinstance(ref_val, list) or isinstance(test_val, list):
            if not isinstance(ref_val, list) or not isinstance(test_val, list):
                return False
            if len(ref_val) != len(test_val):
                return False
            return all(
                self._is_within_tolerance(field, r, t)
                for r, t in zip(sorted(ref_val), sorted(test_val))
            )
        try:
            ref_num, test_num = float(ref_val), float(test_val)
        except (ValueError, TypeError):
            return ref_val == test_val
        tolerance = field.similarity_threshold
        if field.similarity_threshold_mode == "relative":
            tolerance = abs(ref_num) * tolerance
        return abs(ref_num - test_num) <= tolerance

    def _compare_values(self, field, ref_val, test_val):
        """
        Compare two values of a field leaf by leaf.

        Args:
            field (SchemaField): Field whose values are compared
            ref_val: Reference value
            test_val: Test value

        Returns:
            dict: match_fraction, match, metrics and per-leaf results
        """
        metrics = self._new_metrics()
        if not field.value_fields:
            if field.match_mode == "keyed_values":
                match = self._is_value_in_range(
                    ref_val, test_val, self.value_error_thresholds
                )
            else:
                match = self._values_match(field, ref_val, test_val)
            if match:
                metrics["true_positives"] += 1
            else:
                metrics["false_positives"] += 1
                metrics["false_negatives"] += 1
            return {
                "match_fraction": 1.0 if match else 0.0,
                "match": match,
                "metrics": metrics,
                "leaves": None,
            }

        leaves = {}
        matched_weight = 0.0
        total_weight = 0.0
        ref_leaves = list(leaf_values(field.value_fields, ref_val))
        test_leaves = {
            path: value
            for _, path, value in leaf_values(field.value_fields, test_val)
        }
        for leaf, path, ref_leaf in ref_leaves:
            test_leaf = test_leaves.get(path)
            # Skip leaves absent from both
            if ref_leaf is None and test_leaf is None:
                continue
            if ref_leaf is None:
                match = False
                metrics["false_positives"] += 1
            elif test_leaf is None:
                match = False
                metrics["false_negatives"] += 1
            else:
                match = self._values_match(leaf, ref_leaf, test_leaf)
                if match:
                    metrics["true_positives"] += 1
                else:
                    metrics["false_positives"] += 1
                    metrics["false_negatives"] += 1
            total_weight += leaf.effective_weight
            if match:
                matched_weight += leaf.effective_weight
            leaves[".".join(path)] = {
                "match": match,
                "reference": ref_leaf,
                "test": test_leaf,
            }
        if total_weight > 0:
            match_fraction = matched_weight / total_weight
        else:
            match_fraction = 1.0 if all(l["match"] for l in leaves.values()) else 0.0
        return {
            "match_fraction": match_fraction,
            "match": match_fraction == 1.0,
            "metrics": metrics,
            "leaves": leaves,
        }

    def _evaluate_scalar_field(self, field, ref_value, test_value, weight):
        """Evaluate a single-value field with exact or numeric matching."""
        metrics = self._new_metrics()
        penalty = 0.0
        if field.match_mode == "numeric":
            value_match = self._values_match(field, ref_value, test_value)
        else:
            value_match = ref_value == test_value
        details = {
            "match": value_match,
            "reference": ref_value,
            "test": test_value,
        }

        if value_match and ref_value is not None:
            metrics["true_positives"] += 1
        elif ref_value is not None and test_value is not None:
            metrics["false_negatives"] += 1
            metrics["false_positives"] += 1
            penalty += weight
        elif ref_value is not None:
            metrics["false_negatives"] += 1
            penalty += weight
        elif test_value is not None:
            metrics["false_positives"] += 1

        return penalty, details, metrics

    def _evaluate_semantic_field(self, field, ref_value, test_value, weight):
        """Evaluate a text field with semantic similarity."""
        metrics = self._new_metrics()
        penalty = 0.0
        threshold = self._field_threshold(field)

        if ref_value is not None and test_value is not None:
            similarity = self._calculate_text_similarity(ref_value, test_value)
            value_match = similarity >= threshold
            details = {
                "match": value_match,
                "reference": ref_value,
                "test": test_value,
                "similarity": similarity,
            }

            if value_match:
                metrics["true_positives"] += 1
            else:
                # Split the weight between false positive and false negative
                metrics["false_negatives"] += 1
                metrics["false_positives"] += 1
                penalty += weight
        else:
            value_match = ref_value == test_value  # Both None is a match, one None is not
            details = {
                "match": value_match,
                "reference": ref_value,
                "test": test_value,
                "similarity": 1.0 if value_match else 0.0,
            }

            if value_match and ref_value is not None:
                metrics["true_positives"] += 1
            elif ref_value is not None:
                metrics["false_negatives"] += 1
                penalty += weight
            elif test_value is not None:
                metrics["false_positives"] += 1

        return penalty, details, metrics

    def _compare_item_lists(self, field, ref_items, test_items):
        """Match two non-empty lists item by item using tiered similarity."""
        metrics = self._new_metrics()

        # Case 1: Both empty
        if not ref_items and not test_items:
            return {
                "match": True,
                "similarity": 1.0,
                "reference": [],
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": [],
                "metrics": metrics,
            }
        # Case 2: One empty, one not
        if not ref_items:
            # All weight goes to false positives
            metrics["false_positives"] += len(test_items) * 2
            return {
                "match": False,
                "similarity": 0.0,
                "reference": [],
                "test": test_items,
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": [],
                "extra_items": test_items,
                "metrics": metrics,
            }
        if not test_items:
            # All weight goes to false negatives
            metrics["false_negatives"] += len(ref_items) * 2
            return {
                "match": False,
                "similarity": 0.0,
                "reference": ref_items,
                "test": [],
                "very_high_similarity_matches": [],
                "high_similarity_matches": [],
                "missing_items": ref_items,
                "extra_items": [],
                "metrics": metrics,
            }

        # Case 3: Both non-empty, evaluate with item list comparison
        # Define similarity thresholds
        very_high_threshold = self._field_threshold(field)
        high_threshold = 0.70

        # Calculate similarity matrix
        similarity_matrix = []
        for ref_item in ref_items:
            row = []
            for test_item in test_items:
                similarity = self._calculate_text_similarity(ref_item, test_item)
                row.append(similarity)
            similarity_matrix.append(row)

        # Track matches
        very_high_matches = []
        high_matches = []
        matched_ref_idxs = set()
        matched_test_idxs = set()

        # First pass: Find very high similarity matches
        for i, ref_item in enumerate(ref_items):
            if i in matched_ref_idxs:
                continue  # Skip already matched reference items

            best_match_idx = -1
            best_similarity = very_high_threshold

            for j, test_item in enumerate(test_items):
                if j in matched_test_idxs:
                    continue  # Skip already matched test items

                if similarity_matrix[i][j] >= best_similarity:
                    best_similarity = similarity_matrix[i][j]
                    best_match_idx = j

            if best_match_idx >= 0:
                test_item = test_items[best_match_idx]
                matched_ref_idxs.add(i)
                matched_test_idxs.add(best_match_idx)

                very_high_matches.append(
                    {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                )

                # Add to metrics - count as true positive
                metrics["true_positives"] += 2  # Count as 2 true positives

        # Second pass: Find high similarity matches (>0.70)
        for i, ref_item in enumerate(ref_items):
            if i in matched_ref_idxs:
                continue  # Skip already matched reference items

            best_match_idx = -1
            best_similarity = high_threshold

            for j, test_item in enumerate(test_items):
                if j in matched_test_idxs:
                    continue  # Skip already matched test items

                if similarity_matrix[i][j] >= best_similarity:
                    best_similarity = similarity_matrix[i][j]
                    best_match_idx = j

            if best_match_idx >= 0:
                test_item = test_items[best_match_idx]
                matched_ref_idxs.add(i)
                matched_test_idxs.add(best_match_idx)

                high_matches.append(
                    {
                        "reference": ref_item,
                        "test": test_item,
                        "similarity": best_similarity,
                    }
                )

                # Only partial credit for high similarity matches
                metrics["true_positives"] += 1
                metrics["false_positives"] += 1

        # Process unmatched items
        missing_items = [
            ref_items[i] for i in range(len(ref_items)) if i not in matched_ref_idxs
        ]
        extra_items = [
            test_items[i] for i in range(len(test_items)) if i not in matched_test_idxs
        ]

        # Add metrics for missing items
        if missing_items:
            metrics["false_negatives"] += len(missing_items) * 2

        # Add metrics for extra items
        if extra_items:
            metrics["false_positives"] += len(extra_items) * 2

        # Calculate overall similarity
        total_matches = len(very_high_matches) + 0.7 * len(high_matches)
        similarity_score = total_matches / len(ref_items) if ref_items else 0.0

        return {
            "match": similarity_score >= 0.80,  # Threshold for overall match
            "similarity": similarity_score,
            "reference": ref_items,
            "test": test_items,
            "very_high_similarity_matches": very_high_matches,
            "high_similarity_matches": high_matches,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "fractional_value": 1.0 / len(ref_items) if ref_items else 0,
            "metrics": metrics,
        }

    def _evaluate_item_list_field(self, field, ref_items, test_items, weight):
        """Evaluate a list field by matching its items."""
        details = self._compare_item_lists(field, ref_items, test_items)
        metrics = details.pop("metrics")
        penalty = 0.0
        if ref_items and not test_items:
            penalty += weight
        elif ref_items and test_items and not details["match"]:
            # Apply score penalty if not a good match
            penalty += weight * (1 - details["similarity"])
        return penalty, details, metrics

    def _evaluate_joined_text_field(self, field, ref_steps, test_steps, weight):
        """Evaluate a list field by comparing its items joined into one paragraph."""
        metrics = self._new_metrics()
        penalty = 0.0
        threshold = self._field_threshold(field)

        try:
            # Case 1: Both empty
            if not ref_steps and not test_steps:
                details = {
                    "steps_match": True,
                    "reference_steps": [],
                    "test_steps": [],
                }
            # Case 2: One empty, one not
            elif not ref_steps:
                details = {
                    "steps_match": False,
                    "message": "Reference steps are empty",
                    "reference_steps": [],
                    "test_steps": test_steps,
                }

                # All weight goes to false positives
                metrics["false_positives"] += 2  # Count as 2 false positives

            elif not test_steps:
                details = {
                    "steps_match": False,
                    "message": "Test steps are empty",
                    "reference_steps": ref_steps,
                    "test_steps": [],
                }

                # All weight goes to false negatives
                metrics["false_negatives"] += 2  # Count as 2 false negatives
                penalty += weight

            else:
                # Case 3: Both non-empty, use semantic paragraph comparison
                ref_paragraph = " ".join(ref_steps)
                test_paragraph = " ".join(test_steps)
                paragraph_similarity = self._calculate_text_similarity(
                    ref_paragraph, test_paragraph
                )

                steps_match = paragraph_similarity >= threshold

                details = {
                    "reference_steps": ref_steps,
                    "test_steps": test_steps,
                    "paragraph_similarity": paragraph_similarity,
                    "steps_match": steps_match,
                    "paragraph_comparison": {
                        "similarity_score": paragraph_similarity,
                        "high_similarity": paragraph_similarity >= threshold,
                        "method": (
                            "semantic_model"
                            if self.model_available
                            else "sequence_matcher"
                        ),
                    },
                }

                # Add metrics based on similarity
                if steps_match:
                    # High similarity - all weight to true positives
                    metrics["true_positives"] += 2
                else:
                    # Low similarity - split between false positive and negative
                    metrics["false_positives"] += 1
                    metrics["false_negatives"] += 1
                    penalty += weight  # Full penalty
        except Exception as e:
            # Handle any errors in steps evaluation
            print(f"Error in step evaluation: {str(e)}")
            details = {
                "steps_match": False,
                "error": str(e),
                "reference_steps": ref_steps,
                "test_steps": test_steps,
            }

            # Count as false positive and negative
            metrics["false_positives"] += 1
            metrics["false_negatives"] += 1
            penalty += weight

        return penalty, details, metrics

    def _evaluate_object_field(self, field, ref_value, test_value, weight):
        """Evaluate a nested object field leaf by leaf."""
        if ref_value is None and test_value is None:
            return 0.0, {"match": True, "reference": None, "test": None}, self._new_metrics()
        comparison = self._compare_values(field, ref_value, test_value)
        details = {
            "match": comparison["match"],
            "similarity": comparison["match_fraction"],
            "reference": ref_value,
            "test": test_value,
            "leaves": comparison["leaves"],
        }
        penalty = weight * (1 - comparison["match_fraction"])
        return penalty, details, comparison["metrics"]

    def _evaluate_keyed_values_field(self, field, ref_values, test_values, weight):
        """Evaluate a composition-keyed field: match keys semantically, then compare values leaf by leaf."""
        metrics = self._new_metrics()
        penalty = 0.0
        key_threshold = self._field_threshold(field)

        # Case 1: Both empty
        if not ref_values and not test_values:
            details = {
                "match": True,
                "details": {"message": f"Both {field.name} are empty"},
            }
            return penalty, details, metrics

        # Case 2: One empty, one not
        if not ref_values:
            details = {
                "match": False,
                "details": {"message": f"Reference {field.name} is empty"},
            }
            metrics["false_positives"] += sum(
                1 + count_value_items(field, v) for v in test_values.values()
            )
            return penalty, details, metrics

        if not test_values:
            details = {
                "match": False,
                "details": {"message": f"Test {field.name} is empty"},
            }
            metrics["false_negatives"] += sum(
                1 + count_value_items(field, v) for v in ref_values.values()
            )
            penalty += weight
            return penalty, details, metrics

        # Case 3: Both non-empty, first find exact matches using sets, then apply similarity matching
        ref_keys_set = set(ref_values.keys())
        test_keys_set = set(test_values.keys())

        # Step 1: Find exact matches using set operations
        common_keys = ref_keys_set.intersection(test_keys_set)
        remaining_ref_keys = ref_keys_set - common_keys
        remaining_test_keys = test_keys_set - common_keys

        # Track key matches and value matches
        key_matches = {}
        value_matches = {}
        value_fractions = {}
        leaf_matches = {}
        pair_matches = []

        def _add_value_metrics(comparison):
            for metric in metrics:
                metrics[metric] += comparison["metrics"][metric]

        # Process exact matches
        for key in common_keys:
            # The key exists in both sets - count as true positive for key
            metrics["true_positives"] += 1

            key_matches[key] = {
                "matched_with": key,
                "similarity": 1.0,
                "match_type": "exact",
            }

            # Check if values match
            ref_val = ref_values[key]
            test_val = test_values[key]
            comparison = self._compare_values(field, ref_val, test_val)
            _add_value_metrics(comparison)
            value_matches[key] = comparison["match"]
            value_fractions[key] = comparison["match_fraction"]
            if comparison["leaves"] is not None:
                leaf_matches[key] = comparison["leaves"]

            pair_matches.append(
                {
                    "match": comparison["match"],
                    "reference": {key: ref_val},
                    "test": {key: test_val},
                    "similarity": 1.0,
                    "match_type": (
                        "exact" if comparison["match"] else "exact_key_different_value"
                    ),
                }
            )

        # Step 2: Apply similarity matching for remaining keys
        # Convert remaining keys to lists for indexing
        remaining_ref_keys_list = list(remaining_ref_keys)
        remaining_test_keys_list = list(remaining_test_keys)

        # Calculate similarity matrix between remaining ref keys and test keys
        similarity_matrix = []
        for ref_key in remaining_ref_keys_list:
            row = []
            for test_key in remaining_test_keys_list:
                similarity = self._calculate_text_similarity(ref_key, test_key)
                row.append(similarity)
            similarity_matrix.append(row)

        # Track which keys have been matched already
        matched_ref_idxs = set()
        matched_test_idxs = set()

        # Find high similarity matches (similarity > key threshold)
        for i, ref_key in enumerate(remaining_ref_keys_list):
            if i in matched_ref_idxs:
                continue  # Skip already matched reference keys

            best_match_idx = -1
            best_similarity = key_threshold

            # Find best match for this reference key
            for j, test_key in enumerate(remaining_test_keys_list):
                if j in matched_test_idxs:
                    continue  # Skip already matched test keys

                if similarity_matrix[i][j] > best_similarity:
                    best_similarity = similarity_matrix[i][j]
                    best_match_idx = j

            # If a high similarity match is found
            if best_match_idx >= 0:
                test_key = remaining_test_keys_list[best_match_idx]
                matched_ref_idxs.add(i)
                matched_test_idxs.add(best_match_idx)

                # Store match information
                key_matches[ref_key] = {
                    "matched_with": test_key,
                    "similarity": best_similarity,
                    "match_type": "high_similarity",
                }

                # For high similarity matches, give partial credit for the key
                metrics["true_positives"] += 1

                ref_val = ref_values[ref_key]
                test_val = test_values[test_key]
                comparison = self._compare_values(field, ref_val, test_val)
                _add_value_metrics(comparison)
                value_matches[ref_key] = comparison["match"]
                value_fractions[ref_key] = comparison["match_fraction"]
                if comparison["leaves"] is not None:
                    leaf_matches[ref_key] = comparison["leaves"]

                pair_matches.append(
                    {
                        "match": False,  # Not a true match because key isn't exact
                        "reference": {ref_key: ref_val},
                        "test": {test_key: test_val},
                        "similarity": best_similarity,
                        "match_type": (
                            "high_similarity_same_value"
                            if comparison["match"]
                            else "high_similarity_different_value"
                        ),
                    }
                )

        # Process unmatched reference keys (truly missing keys)
        unmatched_ref_keys = [
            remaining_ref_keys_list[i]
            for i in range(len(remaining_ref_keys_list))
            if i not in matched_ref_idxs
        ]

        if unmatched_ref_keys:
            # Missing keys - count as false negatives for the key and every value item
            metrics["false_negatives"] += sum(
                1 + count_value_items(field, ref_values[key])
                for key in unmatched_ref_keys
            )

            for key in unmatched_ref_keys:
                pair_matches.append(
                    {
                        "match": False,
                        "reference": {key: ref_values[key]},
                        "test": None,
                        "similarity": 0.0,
                        "match_type": "missing",
                    }
                )

        # Process unmatched test keys (truly extra keys)
        unmatched_test_keys = [
            remaining_test_keys_list[i]
            for i in range(len(remaining_test_keys_list))
            if i not in matched_test_idxs
        ]

        if unmatched_test_keys:
            # Extra keys - count as false positives for the key and every value item
            metrics["false_positives"] += sum(
                1 + count_value_items(field, test_values[key])
                for key in unmatched_test_keys
            )

            for key in unmatched_test_keys:
                pair_matches.append(
                    {
                        "match": False,
                        "reference": None,
                        "test": {key: test_values[key]},
                        "similarity": 0.0,
                        "match_type": "extra",
                    }
                )

        # Calculate match ratios
        total_ref_keys = len(ref_keys_set)

        # Exact matches are from common_keys
        exact_matches = len(common_keys)

        # High similarity matches
        high_similarity_matches = len(matched_ref_idxs)

        # Key match ratio considers both exact and high similarity matches
        matched_keys_count = exact_matches + high_similarity_matches
        key_match_ratio = (
            matched_keys_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Only consider exact matched keys for value match ratio
        value_match_count = sum(value_fractions[k] for k in common_keys)
        value_match_ratio = (
            value_match_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Pair matches - exact key matches, credited by the fraction of the value that matches
        pair_match_count = sum(value_fractions[k] for k in common_keys)
        pair_match_ratio = (
            pair_match_count / total_ref_keys if total_ref_keys > 0 else 0
        )

        # Overall match calculation
        overall_match_ratio = 0.4 * key_match_ratio + 0.6 * pair_match_ratio
        composition_match = (
            overall_match_ratio > 0.85
        )  # Using 85% threshold for overall match

        # Apply penalty if not a good match
        if not composition_match:
            penalty += weight * (1 - overall_match_ratio)

        # Build detailed results
        value_details = {
            "key_matches": key_matches,
            "value_matches": value_matches,
            "pair_matches": pair_matches,
            "key_match_ratio": key_match_ratio,
            "value_match_ratio": value_match_ratio,
            "pair_match_ratio": pair_match_ratio,
            "overall_match_ratio": overall_match_ratio,
            "exact_matches": exact_matches,
            "high_similarity_matches": high_similarity_matches,
            "missing_keys": unmatched_ref_keys,
            "extra_keys": unmatched_test_keys,
            "similarity_threshold_used": key_threshold,
            "fractional_value": 1.0 / total_ref_keys if total_ref_keys > 0 else 0,
            "total_reference_items": total_ref_keys,
        }
        if leaf_matches:
            value_details["leaf_matches"] = leaf_matches

        details = {
            "match": composition_match,
            "similarity_score": overall_match_ratio,
            "details": value_details,
        }

        return penalty, details, metrics

    def _count_all_items(self, item):
        """Count all individual items in a data entry"""
        return count_record_items(self.schema_config, item)

    def evaluate(
        self,
        ground_truth_file=None,
        test_data_file=None,
        extraction_agent_model_name=None,
        weights=None,
        output_file=DefaultPaths.DETAILED_EVALUATION_FILENAME,
        is_synthesis_evaluation=True,
        value_error_thresholds=None,
    ):
        """
        Evaluate materials science data using normalized weights to ensure fair comparison
        across DOIs with different numbers of items.

        Args:
            ground_truth_file (str): Path to the ground truth JSON file
            test_data_file (str): Path to the test data JSON file
            extraction_agent_model_name (str): Name of the agent model used in data extraction
            weights (dict, optional): Custom weights for different components
            output_file (str, optional): Path to save the detailed evaluation results
            is_synthesis_evaluation (bool, optional): Whether to evaluate synthesis data
            value_error_thresholds (dict, optional): Per-call override for the instance-level
                ``value_error_thresholds``.  Mapping of ``(min, max)`` tuples to absolute
                error tolerances for numeric property value comparisons (see class docstring).

        Returns:
            dict: Evaluation results with scores and details including F1 metrics
        """
        if value_error_thresholds is not None:
            self.value_error_thresholds = value_error_thresholds
        if not ground_truth_file or not test_data_file:
            raise ValueErrorHandler(
                "Both ground truth and test data files are required"
            )

        if not extraction_agent_model_name:
            raise ValueErrorHandler("Used agent model name is required")

        if weights is None:
            weights = self.default_weights

        # Load the JSON files
        with open(ground_truth_file, "r", encoding="utf-8") as gt_file:
            reference_data_full = json.load(gt_file)

        with open(test_data_file, "r", encoding="utf-8") as test_file:
            test_data_full = json.load(test_file)

        # Initialize results
        accumulated_results = {
            "extraction_agent_model_name": extraction_agent_model_name,
            "overall_accuracy": 0.0,
            "overall_composition_accuracy": 0.0,
            "overall_synthesis_accuracy": 0.0,
            "total_items": 0,
            "absolute_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "normalized_classification_metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "item_results": {},
        }

        all_dois = set(reference_data_full.keys()).union(set(test_data_full.keys()))
        total_dois = len(all_dois)  # Total number of unique DOIs

        if total_dois == 0:
            # No data to evaluate
            print("No DOIs found in either reference or test data")
            return accumulated_results

        accumulated_results["total_items"] = total_dois

        # Track scores for calculating overall accuracy
        total_scores = 0.0
        total_composition_score = 0.0
        total_synthesis_score = 0.0

        # Process all DOIs
        for doi in all_dois:
            # Initialize DOI-specific metrics
            item_results = {
                "field_scores": {},
                "absolute_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "normalized_classification_metrics": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "details": {},
            }

            # Case 1: DOI exists in both datasets
            if doi in reference_data_full and doi in test_data_full:
                ref_item = reference_data_full[doi]
                test_item = test_data_full[doi]

                # Evaluate composition data
                comp_score, comp_details, comp_abs_metrics, comp_weights = (
                    self._evaluate_composition_data(
                        ref_item.get("composition_data", {}),
                        test_item.get("composition_data", {}),
                        weights,
                    )
                )

                item_results["field_scores"]["composition_data"] = comp_score
                item_results["details"]["composition_data"] = comp_details
                total_composition_score += comp_score

                # Add composition metrics to article metrics
                for metric in ["true_positives", "false_positives", "false_negatives"]:
                    item_results["absolute_classification_metrics"][
                        metric
                    ] += comp_abs_metrics[metric]

                # Evaluate synthesis data if requested
                if is_synthesis_evaluation:
                    synth_score, synth_details, synth_abs_metrics, synth_weights = (
                        self._evaluate_synthesis_data(
                            ref_item.get("synthesis_data", {}),
                            test_item.get("synthesis_data", {}),
                            weights,
                        )
                    )

                    item_results["field_scores"]["synthesis_data"] = synth_score
                    item_results["details"]["synthesis_data"] = synth_details
                    total_synthesis_score += synth_score

                    # Add synthesis metrics to article metrics
                    for metric in [
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    ]:
                        item_results["absolute_classification_metrics"][
                            metric
                        ] += synth_abs_metrics[metric]

                    # Calculate overall score as weighted average
                    overall_score = 0.5 * comp_score + 0.5 * synth_score
                else:
                    # If only composition evaluation, that's the overall score
                    overall_score = comp_score

                total_scores += overall_score
                item_results["overall_score"] = overall_score
                item_results["overall_match"] = (
                    overall_score >= 0.8
                )  # 80% threshold for match

                # Calculate normalized classification metrics where each DOI contributes exactly 1.0 to the total
                tp_abs = item_results["absolute_classification_metrics"][
                    "true_positives"
                ]
                fp_abs = item_results["absolute_classification_metrics"][
                    "false_positives"
                ]
                fn_abs = item_results["absolute_classification_metrics"][
                    "false_negatives"
                ]

                # Total items for this DOI
                total_items = tp_abs + fp_abs + fn_abs

                if total_items > 0:
                    # Normalize to 1.0 contribution per DOI
                    item_results["normalized_classification_metrics"][
                        "true_positives"
                    ] = (tp_abs / total_items)
                    item_results["normalized_classification_metrics"][
                        "false_positives"
                    ] = (fp_abs / total_items)
                    item_results["normalized_classification_metrics"][
                        "false_negatives"
                    ] = (fn_abs / total_items)

                # Calculate absolute metrics
                if tp_abs + fp_abs > 0:
                    item_results["absolute_classification_metrics"]["precision"] = (
                        tp_abs / (tp_abs + fp_abs)
                    )
                if tp_abs + fn_abs > 0:
                    item_results["absolute_classification_metrics"]["recall"] = (
                        tp_abs / (tp_abs + fn_abs)
                    )

                precision_abs = item_results["absolute_classification_metrics"][
                    "precision"
                ]
                recall_abs = item_results["absolute_classification_metrics"]["recall"]

                if precision_abs + recall_abs > 0:
                    item_results["absolute_classification_metrics"]["f1_score"] = (
                        2 * precision_abs * recall_abs / (precision_abs + recall_abs)
                    )

                # Calculate normalized metrics
                tp_norm = item_results["normalized_classification_metrics"][
                    "true_positives"
                ]
                fp_norm = item_results["normalized_classification_metrics"][
                    "false_positives"
                ]
                fn_norm = item_results["normalized_classification_metrics"][
                    "false_negatives"
                ]

                if tp_norm + fp_norm > 0:
                    item_results["normalized_classification_metrics"]["precision"] = (
                        tp_norm / (tp_norm + fp_norm)
                    )
                if tp_norm + fn_norm > 0:
                    item_results["normalized_classification_metrics"]["recall"] = (
                        tp_norm / (tp_norm + fn_norm)
                    )

                precision_norm = item_results["normalized_classification_metrics"][
                    "precision"
                ]
                recall_norm = item_results["normalized_classification_metrics"][
                    "recall"
                ]

                if precision_norm + recall_norm > 0:
                    item_results["normalized_classification_metrics"]["f1_score"] = (
                        2
                        * precision_norm
                        * recall_norm
                        / (precision_norm + recall_norm)
                    )

            # Case 2: DOI exists in reference but not in test (missing)
            elif doi in reference_data_full:
                ref_item = reference_data_full[doi]
                item_count = self._count_all_items(ref_item)

                # All items are false negatives
                item_results["absolute_classification_metrics"][
                    "false_negatives"
                ] = item_count
                item_results["normalized_classification_metrics"][
                    "false_negatives"
                ] = 1.0  # Full DOI weight
                item_results["overall_score"] = 0.0
                item_results["overall_match"] = False

                # Zero contribution to total scores
                if is_synthesis_evaluation:
                    item_results["field_scores"]["composition_data"] = 0.0
                    item_results["field_scores"]["synthesis_data"] = 0.0
                else:
                    item_results["field_scores"]["composition_data"] = 0.0

            # Case 3: DOI exists in test but not in reference (extra)
            elif doi in test_data_full:
                test_item = test_data_full[doi]
                item_count = self._count_all_items(test_item)

                # All items are false positives
                item_results["absolute_classification_metrics"][
                    "false_positives"
                ] = item_count
                item_results["normalized_classification_metrics"][
                    "false_positives"
                ] = 1.0  # Full DOI weight
                item_results["overall_score"] = 0.0
                item_results["overall_match"] = False

                # No contribution to total scores

            accumulated_results["item_results"][doi] = item_results

            # Add to accumulated metrics
            for metric in ["true_positives", "false_positives", "false_negatives"]:
                accumulated_results["absolute_classification_metrics"][
                    metric
                ] += item_results["absolute_classification_metrics"][metric]
                accumulated_results["normalized_classification_metrics"][
                    metric
                ] += item_results["normalized_classification_metrics"][metric]

        # Calculate overall metrics
        tp_abs = accumulated_results["absolute_classification_metrics"][
            "true_positives"
        ]
        fp_abs = accumulated_results["absolute_classification_metrics"][
            "false_positives"
        ]
        fn_abs = accumulated_results["absolute_classification_metrics"][
            "false_negatives"
        ]

        if tp_abs + fp_abs > 0:
            accumulated_results["absolute_classification_metrics"]["precision"] = (
                tp_abs / (tp_abs + fp_abs)
            )
        if tp_abs + fn_abs > 0:
            accumulated_results["absolute_classification_metrics"]["recall"] = (
                tp_abs / (tp_abs + fn_abs)
            )

        precision_abs = accumulated_results["absolute_classification_metrics"][
            "precision"
        ]
        recall_abs = accumulated_results["absolute_classification_metrics"]["recall"]

        if precision_abs + recall_abs > 0:
            accumulated_results["absolute_classification_metrics"]["f1_score"] = (
                2 * precision_abs * recall_abs / (precision_abs + recall_abs)
            )

        # Calculate normalized metrics
        tp_norm = accumulated_results["normalized_classification_metrics"][
            "true_positives"
        ]
        fp_norm = accumulated_results["normalized_classification_metrics"][
            "false_positives"
        ]
        fn_norm = accumulated_results["normalized_classification_metrics"][
            "false_negatives"
        ]

        if tp_norm + fp_norm > 0:
            accumulated_results["normalized_classification_metrics"]["precision"] = (
                tp_norm / (tp_norm + fp_norm)
            )
        if tp_norm + fn_norm > 0:
            accumulated_results["normalized_classification_metrics"]["recall"] = (
                tp_norm / (tp_norm + fn_norm)
            )

        precision_norm = accumulated_results["normalized_classification_metrics"][
            "precision"
        ]
        recall_norm = accumulated_results["normalized_classification_metrics"]["recall"]

        if precision_norm + recall_norm > 0:
            accumulated_results["normalized_classification_metrics"]["f1_score"] = (
                2 * precision_norm * recall_norm / (precision_norm + recall_norm)
            )

        # Calculate overall accuracy
        if total_dois > 0:
            accumulated_results["overall_accuracy"] = total_scores / total_dois
            accumulated_results["overall_composition_accuracy"] = (
                total_composition_score / total_dois
            )
            if is_synthesis_evaluation:
                accumulated_results["overall_synthesis_accuracy"] = (
                    total_synthesis_score / total_dois
                )

        # Save results to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(accumulated_results, file, indent=2)

        return accumulated_results
