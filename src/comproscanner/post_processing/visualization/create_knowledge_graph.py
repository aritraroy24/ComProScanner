"""
create_knowledge_graph.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 08-04-2025
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import json
from typing import Dict, Optional, List
from dotenv import load_dotenv
import os
from tqdm import tqdm
import sys
import difflib
from collections import Counter

# Try to import advanced NLP models for better semantic similarity
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

# Import utility functions
try:
    from ...utils.configs.schema_config import load_schema_config
    from ...utils.logger import setup_logger
    from ...utils.get_paper_data import PaperMetadataExtractor
    from ...utils.error_handler import ValueErrorHandler
except ImportError:
    print("Error importing utility modules. Make sure they are in the correct path.")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Setup logger
logger = setup_logger("comproscanner.log", module_name="create_knowledge_graph")


class SemanticMatcher:
    """Class for semantic matching of text strings"""

    def __init__(self, model_name="thellert/physbert_cased"):
        """
        Initialize the semantic matching with the specified model

        Args:
            model_name (str): Name of the transformer model to load
        """
        self.semantic_model = None
        self._load_semantic_model(model_name)

    def _load_semantic_model(self, model_name):
        """
        Load the specified semantic model for similarity calculations.

        Args:
            model_name (str): Name of the model to load

        Returns:
            dict: Dictionary with model type and model/tokenizer objects
        """
        # Try loading the transformer model first
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.debug(f"Attempting to load {model_name} transformer model...")
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
                logger.warning(f"Could not load {model_name}: {e}")

        # Try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Falling back to sentence-transformers model...")
                st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.semantic_model = {
                    "type": "sentence_transformer",
                    "model": st_model,
                }
                logger.info("Successfully loaded sentence-transformers model")
                return self.semantic_model
            except Exception as e:
                logger.warning(f"Could not load sentence-transformers: {e}")

        # Final fallback to difflib
        logger.info(
            "Falling back to difflib.SequenceMatcher for similarity calculations"
        )
        self.semantic_model = {"type": "difflib"}
        return self.semantic_model

    def calculate_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts using the best available method.

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
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
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
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
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
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
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def cluster_items(self, items, similarity_threshold=0.8):
        """
        Cluster similar items using semantic similarity.

        Args:
            items (list): List of items to cluster
            similarity_threshold (float): Minimum similarity to consider two items as same

        Returns:
            dict: Dictionary mapping canonical names to lists of similar items
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

    def get_canonical_name(self, item, clusters):
        """
        Get the canonical name for an item from the clusters.

        Args:
            item (str): The item to look up
            clusters (dict): Dictionary mapping canonical names to lists of similar items

        Returns:
            str: The canonical name for the item, or the item itself if not found
        """
        for canonical, similar_items in clusters.items():
            if item in similar_items:
                return canonical
        return item


class CreateKG:
    def __init__(self, schema_config_path=None):
        """
        Initialize Neo4j connection using environment variables

        Args:
            schema_config_path (str | Path, optional): Path to the extraction schema YAML file. Defaults to None (bundled default schema).
        """
        self.driver = None
        self.semantic_matcher = None
        self.schema_config = load_schema_config(schema_config_path)
        self.field_clusters = {}
        self.keyword_clusters = {}

        # Schema fields by (domain, path of field names)
        self._field_by_path = {}

        def index(domain, fields, path):
            for field in fields:
                self._field_by_path[(domain, path + (field.name,))] = field
                if field.value_fields:
                    index(domain, field.value_fields, path + (field.name,))

        for domain in ("composition", "synthesis"):
            index(domain, self.schema_config.fields(domain, final_only=True), ())
        self._primary_field = self._field_by_path[
            ("composition", (self.schema_config.primary_composition_field,))
        ]

        try:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            database = os.getenv(
                "NEO4J_DATABASE", "neo4j"
            )  # Default to "neo4j" if not set

            if not all([uri, user, password]):
                raise ValueError(
                    "Missing required environment variables. Please check your .env file."
                )

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = database
            logger.info(f"Successfully connected to Neo4j database: {database}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except ValueError as e:
            logger.error(str(e))
            raise

    def _initialize_semantic_matcher(self, model_name="thellert/physbert_cased"):
        """Initialize semantic matcher only when needed"""
        if self.semantic_matcher is None:
            logger.info("Initializing semantic matcher for clustering...")
            self.semantic_matcher = SemanticMatcher(model_name)

    def close(self):
        """Close the driver connection"""
        if self.driver is not None:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_canonical_name_safe(self, item, clusters):
        """
        Safely get the canonical name for an item from the clusters.
        If clustering is disabled, return the item as-is.

        Args:
            item (str): The item to look up
            clusters (dict): Dictionary mapping canonical names to lists of similar items

        Returns:
            str: The canonical name for the item, or the item itself if not found
        """
        if self.semantic_matcher is None or not clusters:
            # If clustering is disabled, return the item as-is
            return item
        return self.semantic_matcher.get_canonical_name(item, clusters)

    def _cluster_fields(self):
        """Yield (domain, path, field) for every node_cluster leaf of the schema."""

        def walk(fields, path):
            for field in fields:
                if field.value_fields:
                    yield from walk(field.value_fields, path + (field,))
                elif field.graph_role == "node_cluster":
                    yield path + (field,)

        for domain in ("composition", "synthesis"):
            for path in walk(self.schema_config.fields(domain, final_only=True), ()):
                yield domain, tuple(f.name for f in path), path

    def _cluster_values(self, results, domain, path):
        """
        Collect the values of a node_cluster leaf across all papers for clustering.

        Args:
            results (dict): Dictionary containing results data
            domain (str): "composition" or "synthesis"
            path (tuple): Schema fields from the top-level field down to the leaf

        Returns:
            list: Values to cluster
        """
        leaf = path[-1]
        sentinel = (
            leaf.graph_default.strip().lower()
            if isinstance(leaf.graph_default, str)
            else None
        )
        values = []
        for _, paper_data in results.items():
            section = paper_data.get(f"{domain}_data") or {}
            candidates = [section]
            for field in path:
                next_candidates = []
                for candidate in candidates:
                    if not isinstance(candidate, dict) or field.name not in candidate:
                        continue
                    value = candidate[field.name]
                    # Keyed fields hold one value per entry
                    if field is not leaf and field.parsed_type.kind == "dict":
                        next_candidates.extend((value or {}).values())
                    else:
                        next_candidates.append(value)
                candidates = next_candidates
            for value in candidates:
                if isinstance(value, list):
                    if value:
                        values.extend(value)
                elif value and not (
                    sentinel and isinstance(value, str) and value.strip().lower() == sentinel
                ):
                    values.append(value)
        return values

    def build_field_clusters(self, results, domain, path, similarity_threshold):
        """
        Build clusters of semantically similar values of a node_cluster field

        Args:
            results (dict): Dictionary containing results data
            domain (str): "composition" or "synthesis"
            path (tuple): Schema fields from the top-level field down to the leaf
            similarity_threshold (float): Minimum similarity to consider two values as similar

        Returns:
            dict: Dictionary mapping canonical values to similar values
        """
        return self.semantic_matcher.cluster_items(
            self._cluster_values(results, domain, path), similarity_threshold
        )

    def _top_level_cluster_path(self, name):
        for domain, names, path in self._cluster_fields():
            if names == (name,):
                return domain, path
        return None

    def build_family_clusters(self, results, similarity_threshold):
        """Build clusters of semantically similar material families"""
        domain, path = self._top_level_cluster_path("family")
        return self.build_field_clusters(results, domain, path, similarity_threshold)

    def build_precursor_clusters(self, results, similarity_threshold):
        """Build clusters of semantically similar precursors"""
        domain, path = self._top_level_cluster_path("precursors")
        return self.build_field_clusters(results, domain, path, similarity_threshold)

    def build_method_clusters(self, results, similarity_threshold):
        """Build clusters of semantically similar synthesis methods"""
        domain, path = self._top_level_cluster_path("method")
        return self.build_field_clusters(results, domain, path, similarity_threshold)

    def build_technique_clusters(self, results, similarity_threshold):
        """Build clusters of semantically similar characterization techniques"""
        domain, path = self._top_level_cluster_path("characterization_techniques")
        return self.build_field_clusters(results, domain, path, similarity_threshold)

    def build_keyword_clusters(self, results, similarity_threshold):
        """
        Build clusters of semantically similar keywords

        Args:
            results (dict): Dictionary containing results data
            similarity_threshold (float): Minimum similarity to consider two keywords as similar

        Returns:
            dict: Dictionary mapping canonical keywords to similar keywords
        """
        # Extract all keywords from results
        all_keywords = []
        for _, paper_data in results.items():
            if (
                "article_metadata" in paper_data
                and "keywords" in paper_data["article_metadata"]
            ):
                keywords = paper_data["article_metadata"]["keywords"]
                if keywords:
                    all_keywords.extend(keywords)

        # Cluster keywords
        keyword_clusters = self.semantic_matcher.cluster_items(
            all_keywords, similarity_threshold
        )

        return keyword_clusters

    def _write_field(self, tx, domain, path, parent_ids, value, relationship_type=None):
        """
        Write one schema field's value under its parent nodes, recursing into nested values.

        Args:
            tx: Open Neo4j transaction
            domain (str): "composition" or "synthesis"
            path (tuple): Names of the fields from the top-level field down to this one
            parent_ids (list): Element ids of the parent nodes
            value: The field's value
            relationship_type (str, optional): Overrides the field's relationship type
        """
        field = self._field_by_path[(domain, path)]
        if not parent_ids:
            return
        relationship = relationship_type or field.relationship_type

        # Nested object: one child node per parent
        if field.value_fields and field.is_object:
            if not isinstance(value, dict):
                return
            for parent_id in parent_ids:
                child_id = tx.run(
                    f"MATCH (n) WHERE elementId(n) = $parent_id "
                    f"MERGE (n)-[:{relationship}]->(c:{field.graph_label}) "
                    f"RETURN elementId(c) AS id",
                    parent_id=parent_id,
                ).single().get("id")
                for sub_field in field.value_fields:
                    self._write_field(
                        tx, domain, path + (sub_field.name,), [child_id], value.get(sub_field.name)
                    )
            return

        # Keyed field: one node per entry
        if field.parsed_type.kind == "dict":
            if not isinstance(value, dict) or not value:
                return
            for parent_id in parent_ids:
                for key, entry in value.items():
                    set_value = (
                        ""
                        if field.value_fields
                        else f"SET c.{field.graph_value_property} = $entry "
                    )
                    entry_id = tx.run(
                        f"MATCH (n) WHERE elementId(n) = $parent_id "
                        f"MERGE (n)-[:{relationship}]->(c:{field.graph_label} {{{field.graph_key_property}: $key}}) "
                        f"{set_value}RETURN elementId(c) AS id",
                        parent_id=parent_id,
                        key=key,
                        entry=None if field.value_fields else entry,
                    ).single().get("id")
                    if field.value_fields:
                        for sub_field in field.value_fields:
                            sub_value = entry.get(sub_field.name) if isinstance(entry, dict) else None
                            self._write_field(
                                tx, domain, path + (sub_field.name,), [entry_id], sub_value
                            )
            return

        if field.graph_role == "skip":
            return

        if field.graph_role == "node_property":
            tx.run(
                f"UNWIND $parent_ids AS parent_id "
                f"MATCH (n) WHERE elementId(n) = parent_id "
                f"SET n.{field.name} = $value",
                parent_ids=parent_ids,
                value=value,
            )
            return

        if field.graph_role == "joined_node":
            text = "\n".join(f"- {item}" for item in value or []) or field.graph_default
            if not text:
                return
            tx.run(
                f"MERGE (j:{field.graph_label} {{{field.graph_value_property}: $text}}) "
                f"WITH j UNWIND $parent_ids AS parent_id "
                f"MATCH (n) WHERE elementId(n) = parent_id "
                f"MERGE (n)-[:{relationship}]->(j)",
                parent_ids=parent_ids,
                text=text,
            )
            return

        # node_cluster: shared node per (canonical) value
        if isinstance(value, list):
            values = value
        else:
            values = [] if value is None else [value]
        clusters = self.field_clusters.get((domain, path), {})
        canonical_values = [self.get_canonical_name_safe(v, clusters) for v in values]
        if not canonical_values:
            return
        tx.run(
            f"UNWIND $values AS value "
            f"MERGE (v:{field.graph_label} {{name: value}}) "
            f"WITH v UNWIND $parent_ids AS parent_id "
            f"MATCH (n) WHERE elementId(n) = parent_id "
            f"MERGE (n)-[:{relationship}]->(v)",
            parent_ids=parent_ids,
            values=canonical_values,
        )

    def _section_value(self, section, field):
        """A top-level field's value, falling back to its graph_default when missing."""
        if field.name in section:
            return section[field.name]
        if field.graph_role == "node_cluster":
            return field.graph_default
        return None

    def create_paper_with_compositions(
        self,
        synthesis_data: Dict,
        composition_data: Dict,
        paper_metadata: Dict,
    ) -> bool:
        """
        Create paper node with authors, affiliations, and compositions, and establish relationships

        Args:
            synthesis_data: Dictionary containing synthesis information
            composition_data: Dictionary containing composition information
            paper_metadata: Dictionary containing paper information

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        synthesis_data = synthesis_data or {}
        composition_data = composition_data or {}
        try:
            with self.driver.session(database=self.database) as session:
                # Verify if paper already exists
                check_query = "MATCH (p:Paper {doi: $doi}) RETURN p"
                result = session.run(check_query, doi=paper_metadata.get("doi"))
                exists = result.single() is not None

                if exists:
                    logger.warning(
                        f"Paper with DOI {paper_metadata.get('doi')} already exists in database. Skipping."
                    )
                    return True  # Consider it a success if it already exists

                # Process keywords to use canonical names
                original_keywords = paper_metadata.get("keywords", [])
                canonical_keywords = []

                for keyword in original_keywords:
                    canonical = self.get_canonical_name_safe(
                        keyword, self.keyword_clusters
                    )
                    canonical_keywords.append(canonical)
                    if canonical != keyword:
                        logger.debug(
                            f"Mapped keyword '{keyword}' to canonical '{canonical}'"
                        )

                # Update with canonical keyword names
                paper_metadata_modified = dict(paper_metadata)
                paper_metadata_modified["keywords"] = canonical_keywords
                params = {
                    "paper_metadata": paper_metadata_modified,
                    "keywords": canonical_keywords,
                }

                # Add a transaction wrapper to ensure all-or-nothing operations
                tx = session.begin_transaction()
                try:
                    main_query = """
                    // Create or find paper node based on unique DOI
                    MERGE (p:Paper {doi: $paper_metadata.doi})
                    SET p.title = $paper_metadata.title,
                        p.journal = $paper_metadata.journal,
                        p.year = $paper_metadata.year,
                        p.isOpenAccess = $paper_metadata.isOpenAccess
                    RETURN elementId(p) AS id
                    """

                    result = tx.run(main_query, **params)
                    paper_node = result.single()

                    if not paper_node:
                        logger.error(
                            f"Failed to create paper node for DOI: {paper_metadata.get('doi')}"
                        )
                        tx.rollback()
                        return False
                    paper_id = paper_node.get("id")

                    if paper_metadata.get("authors"):
                        authors_query = """
                        MATCH (p:Paper {doi: $paper_metadata.doi})

                        WITH p
                        UNWIND $paper_metadata.authors as author
                        MERGE (a:Author {name: author.name})
                        SET a.affiliation_id = author.affiliation_id

                        MERGE (aff:Affiliation {affiliation_id: author.affiliation_id})
                        SET aff.name = author.affiliation_name,
                            aff.country = author.affiliation_country

                        MERGE (a)-[:AFFILIATED_WITH]->(aff)
                        MERGE (a)-[:WROTE]->(p)
                        """

                        tx.run(authors_query, **params)

                    # Synthesis fields hang off the paper
                    for field in self.schema_config.fields("synthesis", final_only=True):
                        self._write_field(
                            tx,
                            "synthesis",
                            (field.name,),
                            [paper_id],
                            self._section_value(synthesis_data, field),
                        )

                    # Composition nodes from the primary composition field, scoped to this paper
                    primary = self._primary_field
                    composition_ids = []
                    entries = composition_data.get(primary.name)
                    if isinstance(entries, dict) and entries:
                        set_value = (
                            ""
                            if primary.value_fields
                            else f"SET c.{primary.graph_value_property} = $compositions[comp_name]\n"
                        )
                        compositions_query = f"""
                        MATCH (p:Paper {{doi: $paper_metadata.doi}})

                        WITH p
                        UNWIND keys($compositions) as comp_name
                        // Create or find composition node scoped to this paper
                        MERGE (c:{primary.graph_label} {{{primary.graph_key_property}: comp_name, doi: $paper_metadata.doi}})
                        {set_value}MERGE (p)-[:{primary.relationship_type}]->(c)
                        RETURN comp_name, elementId(c) AS id
                        """
                        records = tx.run(
                            compositions_query,
                            compositions=entries if not primary.value_fields else {k: None for k in entries},
                            **params,
                        ).data()
                        entry_ids = {record["comp_name"]: record["id"] for record in records}
                        composition_ids = list(entry_ids.values())
                        if primary.value_fields:
                            for comp_name, entry_id in entry_ids.items():
                                entry = entries.get(comp_name)
                                for sub_field in primary.value_fields:
                                    sub_value = entry.get(sub_field.name) if isinstance(entry, dict) else None
                                    self._write_field(
                                        tx,
                                        "composition",
                                        (primary.name, sub_field.name),
                                        [entry_id],
                                        sub_value,
                                    )

                    # Other composition fields hang off each composition (and the paper if configured)
                    for field in self.schema_config.fields("composition", final_only=True):
                        if field.name == primary.name:
                            continue
                        value = self._section_value(composition_data, field)
                        if field.graph_edge_sources:
                            for source, relationship_type in field.graph_edge_sources.items():
                                parent_ids = [paper_id] if source == "paper" else composition_ids
                                self._write_field(
                                    tx, "composition", (field.name,), parent_ids, value, relationship_type
                                )
                        else:
                            self._write_field(
                                tx, "composition", (field.name,), composition_ids, value
                            )

                    # Handle keywords separately
                    if canonical_keywords:
                        keywords_query = """
                        MATCH (p:Paper {doi: $paper_metadata.doi})

                        WITH p
                        UNWIND $keywords as keyword_name
                        MERGE (k:Keyword {name: keyword_name})
                        MERGE (p)-[:HAS_KEYWORD]->(k)
                        """

                        tx.run(keywords_query, **params)

                    # Commit the transaction
                    tx.commit()

                    # Verify the paper was actually created
                    verify_query = "MATCH (p:Paper {doi: $doi}) RETURN p"
                    result = session.run(verify_query, doi=paper_metadata.get("doi"))
                    verified = result.single() is not None

                    if verified:
                        return True
                    else:
                        logger.error(
                            f"VERIFICATION FAILED: Paper node for DOI: {paper_metadata.get('doi')} was not found after creation."
                        )
                        return False

                except Exception as e:
                    # If any error occurs, roll back the transaction
                    logger.error(
                        f"Transaction error for DOI {paper_metadata.get('doi')}: {e}"
                    )
                    if tx.closed() == False:
                        tx.rollback()
                    return False

                return True

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return False

    def process_paper_data(
        self,
        synthesis_data: Dict,
        composition_data: Dict,
        paper_metadata: Dict,
    ) -> bool:
        """Process paper data and create nodes in Neo4j database"""
        try:
            success = self.create_paper_with_compositions(
                synthesis_data, composition_data, paper_metadata
            )
            if success:
                return success
            else:
                logger.error(
                    f"Failed to create paper with authors and compositions for DOI: {paper_metadata.get('doi')}"
                )
                return False
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return False

    def create_knowledge_graph(
        self,
        result_file: str = None,
        is_semantic_clustering_enabled: bool = True,
        family_clustering_similarity_threshold: float = 0.9,
        method_clustering_similarity_threshold: float = 0.8,
        precursor_clustering_similarity_threshold: float = 0.9,
        technique_clustering_similarity_threshold: float = 0.8,
        keyword_clustering_similarity_threshold: float = 0.85,
    ):
        """
        Create a knowledge graph from the extracted results file

        Args:
            result_file (str, required): Path to the JSON file containing extracted results.
            is_semantic_clustering_enabled (bool, optional): Whether to enable clustering of similar compositions (Default: True)
            family_clustering_similarity_threshold (float, optional): Similarity threshold for family clustering (Default: 0.9)
            method_clustering_similarity_threshold (float, optional): Similarity threshold for method clustering (Default: 0.8)
            precursor_clustering_similarity_threshold (float, optional): Similarity threshold for precursor clustering (Default: 0.9)
            technique_clustering_similarity_threshold (float, optional): Similarity threshold for technique clustering (Default: 0.8)
            keyword_clustering_similarity_threshold (float, optional): Similarity threshold for keyword clustering (Default: 0.85)
        """
        if not result_file:
            raise ValueErrorHandler(
                "result_file is required to create the knowledge graph"
            )

        # Define a local function to load results file
        def load_results_file(file_path: str) -> Dict:
            """
            Load extracted results from a JSON file

            Args:
                file_path: Path to the results JSON file

            Returns:
                Dict: Dictionary containing the extracted results
            """
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading results file {file_path}: {e}")
                raise

        try:
            # Load the extracted results
            logger.debug(f"Loading results from {result_file}")
            results = load_results_file(result_file)

            # Build semantic clusters for methods, techniques, and keywords first
            if is_semantic_clustering_enabled:
                logger.debug(f"Starting semantic clustering phase...")
                # Initialize the semantic matcher
                self._initialize_semantic_matcher()

                # Legacy threshold parameters take precedence for their fields
                legacy_thresholds = {
                    "family": family_clustering_similarity_threshold,
                    "method": method_clustering_similarity_threshold,
                    "precursors": precursor_clustering_similarity_threshold,
                    "characterization_techniques": technique_clustering_similarity_threshold,
                }
                self.field_clusters = {}
                for domain, names, path in self._cluster_fields():
                    threshold = path[-1].cluster_similarity_threshold
                    if len(names) == 1 and names[0] in legacy_thresholds:
                        threshold = legacy_thresholds[names[0]]
                    self.field_clusters[(domain, names)] = self.build_field_clusters(
                        results, domain, path, threshold
                    )
                self.keyword_clusters = self.build_keyword_clusters(
                    results,
                    similarity_threshold=keyword_clustering_similarity_threshold,
                )
                logger.debug("Semantic clustering completed.")
            else:
                logger.debug(
                    "Skipping semantic clustering (is_semantic_clustering_enabled=False)"
                )
                # Initialize empty clusters - items will be used as-is
                self.field_clusters = {}
                self.keyword_clusters = {}

            # Initialize the paper metadata extractor
            paper_metadata_extractor = PaperMetadataExtractor()

            # Process each paper in the results
            logger.info(f"Processing {len(results)} papers")

            success_count = 0
            skipped_count = 0
            failed_count = 0

            for doi, paper_data in tqdm(
                results.items(), desc="Building knowledge graph"
            ):
                # Extract composition and synthesis data
                composition_data = paper_data.get("composition_data", {})
                synthesis_data = paper_data.get("synthesis_data", {})

                # Try to get paper metadata from existing data or fetch if not available
                paper_metadata = paper_data.get("article_metadata", {})

                if not paper_metadata:
                    try:
                        # Try to get paper metadata from external source
                        paper_metadata = paper_metadata_extractor.get_article_metadata(
                            doi
                        )
                    except Exception as e:
                        logger.error(
                            f"Error fetching paper metadata for DOI: {doi}. {e}"
                        )
                        # Use minimal metadata with just the DOI if fetching fails
                        paper_metadata = {"doi": doi}

                # Make sure a DOI is there in the metadata
                if "doi" not in paper_metadata:
                    paper_metadata["doi"] = doi

                # Validate required data
                if not paper_metadata.get("doi"):
                    logger.error(f"Missing DOI for paper. Skipping.")
                    failed_count += 1
                    continue

                # Process the paper data
                success = self.process_paper_data(
                    synthesis_data, composition_data, paper_metadata
                )

                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    logger.error(f"Failed to process paper with DOI: {doi}")

            # Summary stats
            total_expected = len(results)
            logger.info(f"Knowledge graph building process completed:")
            logger.info(f"  Total papers in input file: {total_expected}")
            logger.info(f"  Successfully processed: {success_count}")
            logger.info(f"  Failed: {failed_count}")

            # Verify actual count in database
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
                db_paper_count = result.single()["paper_count"]
                logger.info(f"  Total papers in database: {db_paper_count}")

                # Count unique clustered nodes per label
                cluster_labels = []
                for _, _, path in self._cluster_fields():
                    if path[-1].graph_label not in cluster_labels:
                        cluster_labels.append(path[-1].graph_label)
                for label in cluster_labels + ["Keyword"]:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    logger.info(
                        f"  Total unique {label} nodes in database: {result.single()['count']}"
                    )

            if db_paper_count != success_count:
                logger.warning(
                    f"Discrepancy detected: {success_count} papers reported as successful but only {db_paper_count} found in database"
                )

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
