"""
data_distribution_visualizers.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import difflib
from collections import Counter
from tqdm import tqdm
from ...utils.configs.schema_config import load_schema_config
from ...utils.logger import setup_logger

logger = setup_logger("comproscanner.log", module_name="data_distribution_visualizers")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DataDistributionVisualizer:
    def __init__(self, schema_config_path=None):
        """
        Initialize the DataDistributionVisualizer class for visualizing material data distributions.

        Args:
            schema_config_path (str | Path, optional): Path to the extraction schema YAML file. Defaults to None (bundled default schema).
        """
        self.data = None
        self.schema_config = load_schema_config(schema_config_path)

    def _load_semantic_model(self, model_name="thellert/physbert_cased"):
        """
        Load the specified semantic model for similarity calculations.

        Args:
            model_name (str, optional): Name of the model to load. Defaults to "thellert/physbert_cased".

        Returns:
            dict: Dictionary with model type and model/tokenizer objects.
        """
        # Check if the model is already loaded
        if hasattr(self, "semantic_model") and self.semantic_model is not None:
            return self.semantic_model

        # Try loading the transformer model first
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.debug(
                    f"\n\nAttempting to load {model_name} transformer model..."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self.semantic_model = {
                    "type": "transformers",
                    "tokenizer": tokenizer,
                    "model": model,
                }
                logger.info(f"Successfully loaded {model_name} transformer model")
                return self.semantic_model
            except Exception as e:
                logger.error(f"Could not load {model_name}: {e}")

        # Try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.debug("Falling back to sentence-transformers model...")
                st_model = SentenceTransformer("all-mpnet-base-v2")
                self.semantic_model = {
                    "type": "sentence_transformer",
                    "model": st_model,
                }
                logger.info("Successfully loaded sentence-transformers model")
                return self.semantic_model
            except Exception as e:
                logger.error(f"Could not load sentence-transformers: {e}")

        # Final fallback to difflib
        logger.debug(
            "Falling back to difflib.SequenceMatcher for similarity calculations"
        )
        self.semantic_model = {"type": "difflib"}
        return self.semantic_model

    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using the best available method.

        Args:
            text1 (str, required): First text.
            text2 (str, required): Second text.

        Returns:
            float: Similarity score between 0 and 1.
        """
        # Make sure a model is loaded
        if not hasattr(self, "semantic_model") or self.semantic_model is None:
            self._load_semantic_model()

        # Handle empty or None inputs
        if not text1 or not text2:
            return 0.0

        # Convert to strings if needed
        text1 = str(text1)
        text2 = str(text2)

        # Use the appropriate similarity calculation method
        if self.semantic_model["type"] == "transformers":
            return self._calculate_similarity_transformers(text1, text2)
        elif self.semantic_model["type"] == "sentence_transformer":
            return self._calculate_similarity_sentence_transformer(text1, text2)
        else:
            # Fallback to difflib
            return self._calculate_similarity_difflib(text1, text2)

    def _calculate_similarity_transformers(self, text1, text2):
        """
        Calculate similarity between two texts using transformers model.

        Args:
            text1 (str, required): First text.
            text2 (str, required): Second text.

        Returns:
            float: Similarity score between 0 and 1.
        """
        tokenizer = self.semantic_model["tokenizer"]
        model = self.semantic_model["model"]

        # Tokenize and get embeddings
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # Use CLS token embedding (first token) as sentence representation
        emb1 = outputs1.last_hidden_state[:, 0, :]
        emb2 = outputs2.last_hidden_state[:, 0, :]

        # Normalize the embeddings
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Calculate cosine similarity
        similarity = torch.mm(emb1, emb2.transpose(0, 1)).item()

        return similarity

    def _calculate_similarity_sentence_transformer(self, text1, text2):
        """
        Calculate similarity between two texts using sentence-transformers.

        Args:
            text1 (str, required): First text.
            text2 (str, required): Second text.

        Returns:
            float: Similarity score between 0 and 1.
        """
        model = self.semantic_model["model"]

        # Get embeddings
        embedding1 = model.encode([text1])[0]
        embedding2 = model.encode([text2])[0]

        # Calculate cosine similarity
        import numpy as np

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Ensure result is in range [0, 1]
        return max(0.0, min(1.0, similarity))

    def _calculate_similarity_difflib(self, text1, text2):
        """
        Calculate similarity between two texts using difflib.SequenceMatcher.

        Args:
            text1 (str, required): First text.
            text2 (str, required): Second text.

        Returns:
            float: Similarity score between 0 and 1.
        """
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def cluster_items(self, items, similarity_threshold=0.8):
        """
        Cluster similar items using semantic similarity.

        Args:
            items (list, required): List of items to cluster.
            similarity_threshold (float, optional): Minimum similarity to consider two items as same. Defaults to 0.8.

        Returns:
            dict: Dictionary mapping canonical names to lists of similar items.
        """
        if not items:
            return {}

        items_counter = Counter(items)
        sorted_items = sorted(items_counter.items(), key=lambda x: x[1], reverse=True)

        # Initialize clusters with the most frequent item as first canonical form
        clusters = {}
        processed = set()

        logger.debug(
            f"\nStarting clustering of {len(sorted_items)} unique items with similarity threshold {similarity_threshold}"
        )

        # Use a progress bar that tracks clusters created, not total items
        pbar = tqdm(desc="Clustering progress", unit=" clusters")

        # Process items from most to least frequent
        try:
            for item, count in sorted_items:
                if item in processed:
                    continue

                # Create a new cluster with this item as canonical form
                canonical = item
                clusters[canonical] = [item]
                processed.add(item)

                # Compare to all remaining items
                for other_item, other_count in sorted_items:
                    if other_item in processed:
                        continue

                    # Calculate similarity
                    similarity = self.calculate_similarity(canonical, other_item)

                    if similarity >= similarity_threshold:
                        clusters[canonical].append(other_item)
                        processed.add(other_item)

                # Update progress: increment by 1 for each cluster created
                cluster_size = len(clusters[canonical])
                pbar.set_postfix_str(
                    f"Cluster size: {cluster_size} | Current: {canonical}"
                )
                pbar.update(1)

        finally:
            pbar.close()

        logger.info(
            f"Clustering completed: {len(clusters)} canonical clusters created from {len(sorted_items)} unique items"
        )
        return clusters

    def _resolve_field(self, field_path):
        """
        Resolve a schema field name or dotted leaf path to its location in the data.

        Args:
            field_path (str, required): Field name (e.g. "family") or dotted leaf path (e.g. "compositions_property_values.d33_data.unit").

        Returns:
            tuple: (data_key, top-level SchemaField, remaining path parts).

        Raises:
            ValueError: If field_path does not match any schema field, or references an unknown nested leaf.
        """
        parts = field_path.split(".")
        for domain in ("composition", "synthesis"):
            for field in self.schema_config.fields(domain, final_only=True):
                if field.name != parts[0]:
                    continue
                # Validate the remaining path against the field's value_fields
                current = field
                for part in parts[1:]:
                    sub_fields = {f.name: f for f in current.value_fields or []}
                    if part not in sub_fields:
                        raise ValueError(f"Unknown schema field path '{field_path}'")
                    current = sub_fields[part]
                return f"{domain}_data", field, parts[1:]
        raise ValueError(f"Unknown schema field '{field_path}'")

    def _field_items(self, field_path, skip_falsy=True):
        """
        Collect all values of a field or nested leaf across the loaded data.

        Args:
            field_path (str, required): Field name or dotted leaf path.
            skip_falsy (bool, optional): Skip empty values (otherwise only None is skipped). Defaults to True.

        Returns:
            list: Values, with list values flattened.
        """
        data_key, field, rest = self._resolve_field(field_path)
        items = []
        for doi, item_data in self.data.items():
            section = item_data.get(data_key)
            if not isinstance(section, dict) or field.name not in section:
                continue
            value = section[field.name]
            # Keyed fields hold one value per composition
            if field.match_mode == "keyed_values" and isinstance(value, dict):
                candidates = list(value.values())
            else:
                candidates = [value]
            for candidate in candidates:
                for part in rest:
                    candidate = candidate.get(part) if isinstance(candidate, dict) else None
                if (not candidate) if skip_falsy else (candidate is None):
                    continue
                if isinstance(candidate, list):
                    items.extend(candidate)
                else:
                    items.append(candidate)
        return items

    def _extract_field_counter(
        self, field_path, is_semantic_clustering_enabled=False, similarity_threshold=0.8
    ):
        """
        Count the values of a field, optionally merging semantically similar values.

        Args:
            field_path (str, required): Field name or dotted leaf path.
            is_semantic_clustering_enabled (bool, optional): Whether to cluster similar values. Defaults to False.
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Counter: Counter with (canonicalized) items.
        """
        all_items = self._field_items(field_path)
        if not is_semantic_clustering_enabled:
            return Counter(all_items)

        if not all_items:
            return Counter()

        # Cluster similar items
        clusters = self.cluster_items(all_items, similarity_threshold)

        # Count canonical items
        canonicalized_counts = Counter()
        raw_counts = Counter(all_items)

        for canonical, similar_items in clusters.items():
            canonicalized_counts[canonical] = sum(
                raw_counts[item] for item in similar_items
            )

        return canonicalized_counts

    def _extract_families_with_clustering(self, similarity_threshold=0.8):
        """
        Extract material families with semantic clustering.

        Args:
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Counter: Counter with canonicalized family names.
        """
        return self._extract_field_counter("family", True, similarity_threshold)

    def _extract_precursors_with_clustering(self, similarity_threshold=0.8):
        """
        Extract precursors with semantic clustering.

        Args:
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Counter: Counter with canonicalized precursor names.
        """
        return self._extract_field_counter("precursors", True, similarity_threshold)

    def _extract_characterization_techniques_with_clustering(
        self, similarity_threshold=0.8
    ):
        """
        Extract characterization techniques with semantic clustering.

        Args:
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Counter: Counter with canonicalized technique names.
        """
        return self._extract_field_counter(
            "characterization_techniques", True, similarity_threshold
        )

    def get_clusters(
        self, data_type, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar items based on semantic similarity.

        Args:
            data_type (str, required): Schema field name or dotted leaf path to cluster ("families" is accepted for "family").
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files. Defaults to None.
            folder_path (str, optional): Path to folder containing JSON data files. Defaults to None.
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical names to lists of similar items.

        Raises:
            ValueError: If data_type does not match any supported schema field.
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        field_path = "family" if data_type == "families" else data_type
        try:
            all_items = self._field_items(field_path)
        except ValueError:
            raise ValueError(f"Unsupported data_type: {data_type}")

        if not all_items:
            return {}

        # Return clusters of similar items
        return self.cluster_items(all_items, similarity_threshold)

    # Keep existing methods for backward compatibility
    def get_family_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar material families.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files. Defaults to None.
            folder_path (str, optional): Path to folder containing JSON data files. Defaults to None.
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical family names to similar names.
        """
        return self.get_clusters(
            "families", data_sources, folder_path, similarity_threshold
        )

    def get_technique_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar characterization techniques.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files. Defaults to None.
            folder_path (str, optional): Path to folder containing JSON data files. Defaults to None.
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical technique names to similar names.
        """
        return self.get_clusters(
            "characterization_techniques",
            data_sources,
            folder_path,
            similarity_threshold,
        )

    def get_precursor_clusters(
        self, data_sources=None, folder_path=None, similarity_threshold=0.8
    ):
        """
        Get clusters of similar precursors.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files. Defaults to None.
            folder_path (str, optional): Path to folder containing JSON data files. Defaults to None.
            similarity_threshold (float, optional): Threshold for similarity-based clustering. Defaults to 0.8.

        Returns:
            Dict[str, List[str]]: Dictionary mapping canonical precursor names to similar names.
        """
        return self.get_clusters(
            "precursors", data_sources, folder_path, similarity_threshold
        )

    def _load_data(self, data_sources=None, folder_path=None):
        """
        Load data from files or dictionaries for visualization.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data. Defaults to None.
            folder_path (str, optional): Path to folder containing JSON data files. Defaults to None.
                Either data_sources or folder_path must be provided.

        Returns:
            dict: Combined data dictionary.

        Raises:
            ValueError: If neither data_sources nor folder_path is provided, or if no valid data found.
        """
        if data_sources is None and folder_path is None:
            raise ValueError("Either data_sources or folder_path must be provided")

        all_data = {}

        # Process folder_path if provided
        if folder_path is not None:
            if not os.path.isdir(folder_path):
                raise ValueError(
                    f"The provided folder path does not exist: {folder_path}"
                )

            # Find all JSON files in the folder
            json_files = []
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    json_files.append(os.path.join(folder_path, file))

            # If no data_sources were provided, use all JSON files from the folder
            if data_sources is None:
                data_sources = json_files
            # If data_sources is a list, append the JSON files from the folder
            elif isinstance(data_sources, list):
                data_sources.extend(json_files)

        # Handle the case when data_sources is a single string (filepath)
        if isinstance(data_sources, str):
            if os.path.isfile(data_sources):
                data_sources = [data_sources]
            else:
                raise ValueError(
                    f"The provided file path does not exist: {data_sources}"
                )

        # Process each data source
        for source in data_sources:
            # Load from dict or file
            if isinstance(source, dict):
                self._merge_data(all_data, source)
            elif isinstance(source, str):
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._merge_data(all_data, data)
                except Exception as e:
                    logger.error(f"Error loading {source}: {e}")
                    continue

        if not all_data:
            raise ValueError("No valid data found")

        self.data = all_data
        return all_data

    def _merge_data(self, target_dict, source_dict):
        """
        Merge source data dictionary into target data dictionary.

        Args:
            target_dict (dict, required): Target dictionary to merge data into.
            source_dict (dict, required): Source dictionary containing data to merge.
        """
        for doi, data in source_dict.items():
            target_dict[doi] = data

    def _extract_families(self):
        """
        Extract family information from the loaded data.

        Returns:
            dict: Counter object with family names and their frequencies
        """
        return self._extract_field_counter("family")

    def _extract_precursors(self):
        """
        Extract precursors information from the loaded data.

        Returns:
            dict: Counter object with precursor names and their frequencies
        """
        return self._extract_field_counter("precursors")

    def _extract_characterization_techniques(self):
        """
        Extract characterization techniques information from the loaded data.

        Returns:
            dict: Counter object with technique names and their frequencies
        """
        return self._extract_field_counter("characterization_techniques")

    def _plot_pie_chart(
        self,
        data_counter,
        title,
        figsize=(12, 8),
        dpi=300,
        output_file=None,
        min_percentage=1.0,
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
    ):
        """
        Create a pie chart visualization of data distribution with percentage labels outside the chart.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            min_percentage (float, optional): Minimum percentage for a category to be shown separately (Default: 1.0)
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate percentages and create 'Others' category if needed
        total = sum(data_counter.values())
        percentages = {k: (v / total) * 100 for k, v in data_counter.items()}

        # Sort by percentage (descending)
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create 'Others' category for items below threshold
        main_items = [(k, v) for k, v in sorted_items if v >= min_percentage]
        others = [(k, v) for k, v in sorted_items if v < min_percentage]

        if others:
            others_sum = sum(v for _, v in others)
            plot_items = main_items + [("Others", others_sum)]
        else:
            plot_items = main_items

        labels = [f"{k} ({v:.1f}%)" for k, v in plot_items]
        values = [v for _, v in plot_items]
        percentage_labels = [
            f"{v:.1f}%" for v in values
        ]  # Just the percentage for outside labels

        # Generate colors using specified palette
        cmap = plt.get_cmap(color_palette)
        colors = cmap(np.linspace(0, 0.9, len(plot_items)))

        # Create pie chart with percentage labels outside
        wedges, texts = ax.pie(
            values,
            labels=percentage_labels,  # Use the percentage labels
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "w", "linewidth": 1},
            autopct=None,  # No internal percentage labels
            labeldistance=1.05,  # Position labels outside the pie
        )

        # Improve label positioning and styling
        for text in texts:
            text.set_fontweight("bold")  # Make labels bold
            text.set_fontsize(label_fontsize)

            # Get the position of the text
            pos = text.get_position()
            x, y = pos

            # Calculate angle to determine positioning
            angle = np.arctan2(y, x)
            # Convert to degrees for easier understanding
            angle_deg = np.degrees(angle)

            # Normalize angle to 0-360 range
            if angle_deg < 0:
                angle_deg += 360

            # More precise alignment based on angle ranges
            if 60 <= angle_deg <= 120:  # Top quadrant (60° to 120°)
                text.set_horizontalalignment("center")
                text.set_verticalalignment("bottom")
            elif 240 <= angle_deg <= 300:  # Bottom quadrant (240° to 300°)
                text.set_horizontalalignment("center")
                text.set_verticalalignment("top")
            elif angle_deg <= 30 or angle_deg >= 330:  # Right side (330° to 30°)
                text.set_horizontalalignment("left")
                text.set_verticalalignment("center")
            elif 150 <= angle_deg <= 210:  # Left side (150° to 210°)
                text.set_horizontalalignment("right")
                text.set_verticalalignment("center")
            else:  # Diagonal positions - use adaptive alignment
                if 30 < angle_deg < 60:  # Top-right
                    text.set_horizontalalignment("left")
                    text.set_verticalalignment("bottom")
                elif 120 < angle_deg < 150:  # Top-left
                    text.set_horizontalalignment("right")
                    text.set_verticalalignment("bottom")
                elif 210 < angle_deg < 240:  # Bottom-left
                    text.set_horizontalalignment("right")
                    text.set_verticalalignment("top")
                elif 300 < angle_deg < 330:  # Bottom-right
                    text.set_horizontalalignment("left")
                    text.set_verticalalignment("top")
                else:  # Fallback for any edge cases
                    if x >= 0:
                        text.set_horizontalalignment("left")
                    else:
                        text.set_horizontalalignment("right")
                    text.set_verticalalignment("center")

        # Add legend
        ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=legend_fontsize,
        )

        # Set title directly on the axes, centered
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect("equal")

        # Adjust layout with title positioning
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.info(f"\nPie chart saved to {output_file}")

        return fig

    def _plot_histogram(
        self,
        data_counter,
        title,
        figsize=(12, 8),
        dpi=300,
        output_file=None,
        max_items=15,
        color_palette="Blues",
        x_label=None,
        y_label="Frequency",
        rotation=45,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
    ):
        """
        Create a histogram visualization of data distribution.

        Args:
            data_counter (Counter): Counter object with data labels and frequencies
            title (str): Title for the plot
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            output_file (str, optional): Path to save the output plot image (Default: None)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: None)
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by frequency (descending)
        sorted_items = sorted(data_counter.items(), key=lambda x: x[1], reverse=True)

        # Limit to max_items if needed
        if max_items and len(sorted_items) > max_items:
            display_items = sorted_items[: max_items - 1]

            # Add 'Others' category with sum of remaining items
            others_sum = sum(v for _, v in sorted_items[max_items - 1 :])
            display_items.append(("Others", others_sum))
        else:
            display_items = sorted_items

        labels = [k for k, _ in display_items]
        values = [v for _, v in display_items]

        # Generate colors using specified palette
        cmap = plt.get_cmap(color_palette)
        colors = cmap(np.linspace(0, 0.9, len(display_items)))

        # Create bars
        bars = ax.bar(
            range(len(display_items)),
            values,
            color=colors,
            edgecolor="black",
            alpha=0.8,
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=value_label_fontsize,
            )

        # Set x-axis labels
        ax.set_xticks(range(len(display_items)))
        ax.set_xticklabels(
            labels, rotation=rotation, ha="right", fontsize=xtick_fontsize
        )

        # Set y-axis tick label font size
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        # Set axis labels
        if x_label:
            ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
        ax.set_ylabel(y_label, fontsize=ylabel_fontsize)

        # Set title
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)

        # Add grid
        if grid_axis and grid_axis.lower() != "none":
            ax.grid(axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha)

        # Adjust layout
        plt.tight_layout()

        # Save figure if output_file is provided
        if output_file:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.info(f"\nHistogram saved to {output_file}")

        return fig

    def plot_categorical_distribution(
        self,
        field,
        chart_type="histogram",
        data_sources=None,
        folder_path=None,
        output_file=None,
        title=None,
        color_palette="Blues",
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
        data_label=None,
        **chart_kwargs,
    ):
        """
        Create a pie chart or histogram of the values of any schema field.

        Args:
            field (str, required): Schema field name or dotted leaf path (e.g. "family", "compositions_property_values.d33_data.unit").
            chart_type (str, optional): "pie" or "histogram" (Default: "histogram")
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            title (str, optional): Title for the plot (Default: derived from the field name)
            color_palette (str, optional): Matplotlib colormap name (Default: "Blues")
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of values (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)
            data_label (str, optional): Name used in the "no data" error message (Default: field name)
            **chart_kwargs: Further styling arguments for the pie chart or histogram

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        label = field.split(".")[-1].replace("_", " ")
        if title is None:
            title = f"Distribution of {label.title()}"

        counter = self._extract_field_counter(
            field, is_semantic_clustering_enabled, similarity_threshold
        )
        if is_semantic_clustering_enabled:
            title = f"{title} (Semantically Clustered)"

        if not counter:
            raise ValueError(
                f"No {data_label or label} data found in the provided data sources"
            )

        # Use default colour palette if not specified
        if color_palette is None:
            color_palette = "Blues"

        if chart_type == "pie":
            return self._plot_pie_chart(
                counter,
                title,
                output_file=output_file,
                color_palette=color_palette,
                **chart_kwargs,
            )
        if chart_type == "histogram":
            chart_kwargs.setdefault("x_label", label.title())
            return self._plot_histogram(
                counter,
                title,
                output_file=output_file,
                color_palette=color_palette,
                **chart_kwargs,
            )
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    def _numeric_values(self, field):
        """
        Collect the numeric values of a field or nested leaf, flattening lists.

        Args:
            field (str, required): Schema field name or dotted leaf path.

        Returns:
            list: Numeric values found for the field, booleans excluded.
        """
        values = []
        for item in self._field_items(field, skip_falsy=False):
            if isinstance(item, bool):
                continue
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return values

    def plot_numeric_distribution(
        self,
        field,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        bins=20,
        title=None,
        x_label=None,
        y_label="Frequency",
        color="steelblue",
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        grid_alpha=0.3,
    ):
        """
        Create a histogram of the numeric values of any schema field.

        Args:
            field (str, required): Schema field name or dotted leaf path (e.g. "compositions_property_values", "compositions_property_values.sintering_temp.value").
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            bins (int, optional): Number of histogram bins (Default: 20)
            title (str, optional): Title for the plot (Default: derived from the field name)
            x_label (str, optional): Label for the x-axis (Default: derived from the field name)
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            color (str, optional): Bar colour (Default: "steelblue")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        label = field.split(".")[-1].replace("_", " ")
        values = self._numeric_values(field)
        if not values:
            raise ValueError(f"No numeric {label} data found in the provided data sources")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(title or f"Distribution of {label.title()} Values", fontsize=title_fontsize)
        ax.set_xlabel(x_label or label.title(), fontsize=xlabel_fontsize)
        ax.set_ylabel(y_label, fontsize=ylabel_fontsize)
        ax.grid(axis="y", linestyle="--", alpha=grid_alpha)
        plt.tight_layout()

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.info(f"Plot saved to {output_file}")

        return fig

    def _plot_kind_fields(self):
        """Dotted paths and plot kinds of every schema field or leaf with a plot_kind."""

        def walk(fields, prefix=""):
            for field in fields:
                path = f"{prefix}{field.name}"
                if field.plot_kind != "none":
                    yield path, field.plot_kind
                if field.value_fields:
                    yield from walk(field.value_fields, f"{path}.")

        for domain in ("composition", "synthesis"):
            yield from walk(self.schema_config.fields(domain, final_only=True))

    def plot_all_distributions(
        self,
        data_sources=None,
        folder_path=None,
        output_folder=None,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Plot every schema field that declares a plot_kind.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_folder (str, optional): Folder to save one "<field>_distribution.png" per plot. If None, plots are not saved
            is_semantic_clustering_enabled (bool, optional): Whether to cluster categorical values (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            Dict[str, matplotlib.figure.Figure]: Figures keyed by field path; fields without data are skipped
        """
        # Load data if not already loaded
        if self.data is None or data_sources is not None or folder_path is not None:
            self._load_data(data_sources, folder_path)

        figures = {}
        for path, plot_kind in self._plot_kind_fields():
            output_file = (
                os.path.join(output_folder, f"{path}_distribution.png")
                if output_folder
                else None
            )
            try:
                if plot_kind == "numeric_distribution":
                    figures[path] = self.plot_numeric_distribution(
                        path, output_file=output_file
                    )
                else:
                    figures[path] = self.plot_categorical_distribution(
                        path,
                        output_file=output_file,
                        is_semantic_clustering_enabled=is_semantic_clustering_enabled,
                        similarity_threshold=similarity_threshold,
                    )
            except ValueError as e:
                logger.warning(f"Skipping plot for {path}: {e}")
        return figures

    def plot_family_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Material Families",
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a pie chart visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of families (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "family",
            chart_type="pie",
            data_label="family",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            min_percentage=min_percentage,
            title=title,
            color_palette=color_palette,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
        )

    def plot_family_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Material Families",
        color_palette="Blues",
        x_label="Material Family",
        y_label="Frequency",
        rotation=45,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a histogram visualization of material families distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Material Family")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of families (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "family",
            chart_type="histogram",
            data_label="family",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            max_items=max_items,
            title=title,
            color_palette=color_palette,
            x_label=x_label,
            y_label=y_label,
            rotation=rotation,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_fontsize=xtick_fontsize,
            ytick_fontsize=ytick_fontsize,
            value_label_fontsize=value_label_fontsize,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_alpha=grid_alpha,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
        )

    def plot_precursors_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Precursors in Materials Synthesis",
        color_palette="Blues",
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a pie chart visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend (Default: 10)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of precursors (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "precursors",
            chart_type="pie",
            data_label="precursors",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            min_percentage=min_percentage,
            title=title,
            color_palette=color_palette,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
        )

    def plot_precursors_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Precursors in Materials Synthesis",
        color_palette=None,
        x_label="Precursor",
        y_label="Frequency",
        rotation=45,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
    ):
        """
        Create a histogram visualization of precursors distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Material Family")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)
            is_semantic_clustering_enabled (bool, optional): Whether to enable semantic clustering of precursors (Default: True)
            similarity_threshold (float, optional): Similarity threshold for clustering (Default: 0.8)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "precursors",
            chart_type="histogram",
            data_label="precursors",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            max_items=max_items,
            title=title,
            color_palette=color_palette,
            x_label=x_label,
            y_label=y_label,
            rotation=rotation,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_fontsize=xtick_fontsize,
            ytick_fontsize=ytick_fontsize,
            value_label_fontsize=value_label_fontsize,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_alpha=grid_alpha,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
        )

    def plot_characterization_techniques_pie_chart(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        min_percentage=1.0,
        title="Distribution of Characterization Techniques",
        color_palette="Blues",
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
        title_fontsize=14,
        label_fontsize=10,
        legend_fontsize=10,
    ):
        """
        Create a pie chart visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files. Either data_sources or folder_path must be provided.
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            min_percentage (float, optional): Minimum percentage for a category to be shown separately
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the pie sections (Default: "Blues")
            is_semantic_clustering_enabled (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True
            title_fontsize (int, optional): Font size for the title (Default: 14)
            label_fontsize (int, optional): Font size for the percentage labels (Default: 10)
            legend_fontsize (int, optional): Font size for the legend labels (Default: 10)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "characterization_techniques",
            chart_type="pie",
            data_label="characterization techniques",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            min_percentage=min_percentage,
            title=title,
            color_palette=color_palette,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
        )

    def plot_characterization_techniques_histogram(
        self,
        data_sources=None,
        folder_path=None,
        output_file=None,
        figsize=(12, 8),
        dpi=300,
        max_items=15,
        title="Frequency Distribution of Characterization Techniques",
        color_palette=None,
        x_label="Characterization Technique",
        y_label="Frequency",
        rotation=45,
        is_semantic_clustering_enabled=True,
        similarity_threshold=0.8,
        title_fontsize=14,
        xlabel_fontsize=12,
        ylabel_fontsize=12,
        xtick_fontsize=10,
        ytick_fontsize=10,
        value_label_fontsize=9,
        grid_axis="y",
        grid_linestyle="--",
        grid_alpha=0.3,
    ):
        """
        Create a histogram visualization of characterization techniques distribution.

        Args:
            data_sources (Union[List[str], List[Dict], str], optional): List of paths to JSON files
                or dictionaries containing materials data
            folder_path (str, optional): Path to folder containing JSON data files
            output_file (str, optional): Path to save the output plot image. If None, the plot is not saved.
            figsize (tuple, optional): Figure size as (width, height) in inches (Default: (12, 8))
            dpi (int, optional): DPI for output image (Default: 300)
            max_items (int, optional): Maximum number of items to display (Default: 15)
            title (str, optional): Title for the plot
            color_palette (str, optional): Matplotlib colormap name for the bars (Default: "Blues")
            x_label (str, optional): Label for the x-axis (Default: "Characterization Technique")
            y_label (str, optional): Label for the y-axis (Default: "Frequency")
            rotation (int, optional): Rotation angle for x-axis labels (Default: 45)
            is_semantic_clustering_enabled (bool): Whether to use semantic similarity for clustering similar techniques
            similarity_threshold (float): Threshold for similarity-based clustering when is_semantic_clustering_enabled is True
            title_fontsize (int, optional): Font size for the title (Default: 14)
            xlabel_fontsize (int, optional): Font size for the x-axis label (Default: 12)
            ylabel_fontsize (int, optional): Font size for the y-axis label (Default: 12)
            xtick_fontsize (int, optional): Font size for the x-axis tick labels (Default: 10)
            ytick_fontsize (int, optional): Font size for the y-axis tick labels (Default: 10)
            value_label_fontsize (int, optional): Font size for the value labels on bars (Default: 9)
            grid_axis (str, optional): Axis for grid lines ('x', 'y', 'both', or None for no grid) (Default: "y")
            grid_linestyle (str, optional): Line style for grid lines (Default: "--")
            grid_alpha (float, optional): Alpha (transparency) for grid lines (Default: 0.3)

        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        return self.plot_categorical_distribution(
            "characterization_techniques",
            chart_type="histogram",
            data_label="characterization techniques",
            data_sources=data_sources,
            folder_path=folder_path,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
            max_items=max_items,
            title=title,
            color_palette=color_palette,
            x_label=x_label,
            y_label=y_label,
            rotation=rotation,
            is_semantic_clustering_enabled=is_semantic_clustering_enabled,
            similarity_threshold=similarity_threshold,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_fontsize=xtick_fontsize,
            ytick_fontsize=ytick_fontsize,
            value_label_fontsize=value_label_fontsize,
            grid_axis=grid_axis,
            grid_linestyle=grid_linestyle,
            grid_alpha=grid_alpha,
        )
