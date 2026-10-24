"""
test_create_knowledge_graph.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 16-07-2025
"""

import pytest
import json
import os
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock
import difflib
from collections import Counter


class TestCreateKnowledgeGraphFunction:
    """Test the main create_knowledge_graph function from data_visualizer"""

    @pytest.fixture
    def sample_results_data(self):
        """Sample test data"""
        return {
            "10.1000/test1": {
                "composition_data": {
                    "family": "Perovskite",
                    "compositions_property_values": {"BaTiO3": 180.0},
                    "property_unit": "pC/N",
                },
                "synthesis_data": {
                    "method": "sol-gel",
                    "steps": ["Mix precursors", "Heat to 600°C"],
                    "precursors": ["Ba(OH)2", "TiO2"],
                    "characterization_techniques": ["XRD", "SEM"],
                },
                "article_metadata": {
                    "doi": "10.1000/test1",
                    "title": "Test Paper",
                    "keywords": ["perovskite", "ferroelectric"],
                },
            }
        }

    @pytest.fixture
    def temp_results_file(self, sample_results_data):
        """Create temporary results file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_results_data, f)
            temp_file = f.name
        yield temp_file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_create_knowledge_graph_with_valid_file(self, temp_results_file):
        """Test create_knowledge_graph function with valid input file"""
        # Mock the CreateKG class and its methods
        with patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg:
            # Configure the mock
            mock_kg_instance = Mock()
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            # Import and call the function
            from comproscanner.data_visualizer import create_knowledge_graph

            create_knowledge_graph(
                result_file=temp_results_file, is_semantic_clustering_enabled=True
            )

            # Verify the function was called correctly
            mock_create_kg.assert_called_once()
            mock_kg_instance.create_knowledge_graph.assert_called_once_with(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=True,
                method_clustering_similarity_threshold=0.8,
                technique_clustering_similarity_threshold=0.8,
                keyword_clustering_similarity_threshold=0.85,
                family_clustering_similarity_threshold=0.9,
                precursor_clustering_similarity_threshold=0.9,
            )

    def test_create_knowledge_graph_with_none_file(self):
        """Test create_knowledge_graph function with None result_file"""
        with (
            patch("comproscanner.data_visualizer.CreateKG"),
            patch("comproscanner.data_visualizer.logger") as mock_logger,
        ):

            from comproscanner.data_visualizer import create_knowledge_graph

            # Should raise an exception for None file
            with pytest.raises(
                Exception
            ):  # ValueErrorHandler becomes Exception in mocking
                create_knowledge_graph(result_file=None)

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_create_knowledge_graph_with_custom_thresholds(self, temp_results_file):
        """Test create_knowledge_graph with custom similarity thresholds"""
        with patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg:
            mock_kg_instance = Mock()
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            from comproscanner.data_visualizer import create_knowledge_graph

            create_knowledge_graph(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=False,
                method_clustering_similarity_threshold=0.9,
                technique_clustering_similarity_threshold=0.7,
                keyword_clustering_similarity_threshold=0.95,
            )

            # Verify custom parameters were passed
            mock_kg_instance.create_knowledge_graph.assert_called_once_with(
                result_file=temp_results_file,
                is_semantic_clustering_enabled=False,
                method_clustering_similarity_threshold=0.9,
                technique_clustering_similarity_threshold=0.7,
                keyword_clustering_similarity_threshold=0.95,
                family_clustering_similarity_threshold=0.9,
                precursor_clustering_similarity_threshold=0.9,
            )

    def test_create_knowledge_graph_exception_handling(self, temp_results_file):
        """Test exception handling in create_knowledge_graph"""
        with (
            patch("comproscanner.data_visualizer.CreateKG") as mock_create_kg,
            patch("comproscanner.data_visualizer.logger") as mock_logger,
        ):

            # Configure mock to raise an exception
            mock_kg_instance = Mock()
            mock_kg_instance.create_knowledge_graph.side_effect = Exception(
                "Test error"
            )
            mock_create_kg.return_value.__enter__.return_value = mock_kg_instance
            mock_create_kg.return_value.__exit__.return_value = None

            from comproscanner.data_visualizer import create_knowledge_graph

            # Should re-raise the exception
            with pytest.raises(Exception):
                create_knowledge_graph(result_file=temp_results_file)

            # Verify error was logged
            mock_logger.error.assert_called_with(
                "Error creating knowledge graph: Test error"
            )


class TestSemanticMatcher:
    """Test semantic matching functionality using direct implementation"""

    def test_difflib_similarity_calculation(self):
        """Test similarity calculation using difflib (fallback method)"""
        # Test identical strings
        similarity = difflib.SequenceMatcher(None, "test", "test").ratio()
        assert similarity == 1.0

        # Test completely different strings
        similarity = difflib.SequenceMatcher(None, "hello", "world").ratio()
        assert 0.0 <= similarity <= 1.0  # Just check it's a valid similarity score

        # Test case-insensitive similar strings
        similarity = difflib.SequenceMatcher(None, "XRD".lower(), "xrd".lower()).ratio()
        assert similarity == 1.0  # Should be identical when lowercased

        # Test partially similar strings
        similarity = difflib.SequenceMatcher(None, "X-ray diffraction", "XRD").ratio()
        assert 0.0 <= similarity <= 1.0  # Valid range

    def test_clustering_logic(self):
        """Test basic clustering logic"""
        items = ["XRD", "X-ray diffraction", "SEM", "scanning electron microscopy"]
        similarity_threshold = 0.8

        # Simple clustering algorithm for testing
        clusters = {}
        processed = set()

        for item in items:
            if item in processed:
                continue

            canonical = item
            cluster_items = [item]
            processed.add(item)

            # Find similar items (simplified logic)
            for other_item in items:
                if other_item != item and other_item not in processed:
                    # Simple similarity check
                    if any(
                        word.lower() in other_item.lower()
                        for word in item.lower().split()
                    ):
                        cluster_items.append(other_item)
                        processed.add(other_item)

            clusters[canonical] = cluster_items

        # Verify clustering results
        assert len(clusters) >= 1
        all_clustered_items = [
            item for cluster in clusters.values() for item in cluster
        ]
        assert set(all_clustered_items) == set(items)

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty string similarity
        similarity = difflib.SequenceMatcher(None, "", "test").ratio()
        assert similarity == 0.0

        # Empty list clustering
        clusters = {}
        items = []
        assert clusters == {}


class TestDataStructures:
    """Test data structure handling and validation"""

    def test_valid_paper_data_structure(self):
        """Test validation of paper data structure"""
        paper_data = {
            "composition_data": {
                "family": "Perovskite",
                "compositions_property_values": {"BaTiO3": 180.0},
                "property_unit": "pC/N",
            },
            "synthesis_data": {
                "method": "sol-gel",
                "precursors": ["Ba(OH)2"],
                "characterization_techniques": ["XRD"],
            },
            "article_metadata": {"doi": "10.1000/test", "title": "Test Paper"},
        }

        # Verify required sections exist
        required_sections = ["composition_data", "synthesis_data", "article_metadata"]
        for section in required_sections:
            assert section in paper_data

        # Verify DOI exists
        assert "doi" in paper_data["article_metadata"]
        assert paper_data["article_metadata"]["doi"] is not None

    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        malformed_data = {
            "composition_data": "not_a_dict",  # Should be dict
            "synthesis_data": None,
            "article_metadata": {"doi": "test"},
        }

        # Test that we can detect malformed data
        assert not isinstance(malformed_data["composition_data"], dict)
        assert malformed_data["synthesis_data"] is None
        assert isinstance(malformed_data["article_metadata"], dict)

    def test_json_file_operations(self):
        """Test JSON file reading and writing"""
        test_data = {
            "10.1000/test": {
                "composition_data": {"family": "Test"},
                "synthesis_data": {"method": "test"},
                "article_metadata": {"doi": "10.1000/test"},
            }
        }

        # Test writing and reading JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Read back the data
            with open(temp_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files"""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json content")
            temp_file = f.name

        try:
            # Should raise JSON decode error
            with pytest.raises(json.JSONDecodeError):
                with open(temp_file, "r") as f:
                    json.load(f)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestEnvironmentConfiguration:
    """Test environment configuration handling"""

    def test_environment_variables_validation(self):
        """Test validation of required environment variables"""
        required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]

        # Test with missing variables
        with patch.dict(os.environ, {}, clear=True):
            for var in required_vars:
                assert os.getenv(var) is None

        # Test with present variables
        test_env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        }

        with patch.dict(os.environ, test_env):
            for var in required_vars:
                assert os.getenv(var) is not None

    def test_default_database_name(self):
        """Test default database name handling"""
        # Test default value
        with patch.dict(os.environ, {}, clear=True):
            database_name = os.getenv("NEO4J_DATABASE", "neo4j")
            assert database_name == "neo4j"

        # Test custom value
        with patch.dict(os.environ, {"NEO4J_DATABASE": "custom_db"}):
            database_name = os.getenv("NEO4J_DATABASE", "neo4j")
            assert database_name == "custom_db"


class TestIntegrationScenarios:
    """Test integration scenarios with mocked dependencies"""

    @pytest.fixture
    def mock_neo4j_setup(self):
        """Set up Neo4j mocks"""
        with patch("neo4j.GraphDatabase") as mock_db:
            mock_driver = Mock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver.close.return_value = None

            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None

            mock_session.run.return_value.single.return_value = {"paper_count": 1}

            mock_db.driver.return_value = mock_driver

            yield {"driver": mock_driver, "session": mock_session}


class TestCompositionNodeDoiScoping:
    """Regression tests: the Composition node MERGE must be scoped by doi,
    not just the composition string, or the same composition text reported
    by two different papers collapses into one node and overwrites its
    property value instead of creating a distinct node per paper."""

    @pytest.fixture
    def mock_kg(self):
        """CreateKG with a mocked Neo4j driver/session/transaction wired for
        a successful create_paper_with_compositions call. Yields the
        instance plus the mock transaction so tests can inspect the Cypher
        queries and params passed to tx.run."""
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        }
        with (
            patch.dict(os.environ, env),
            patch(
                "comproscanner.post_processing.visualization.create_knowledge_graph.GraphDatabase"
            ) as mock_graph_db,
        ):
            mock_driver = Mock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver.session.return_value = MagicMock()
            mock_graph_db.driver.return_value = mock_driver

            mock_session = Mock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None

            # Paper doesn't exist on the pre-check, exists after commit
            mock_session.run.return_value.single.side_effect = [None, Mock()]

            mock_tx = Mock()
            mock_tx.run.return_value.single.return_value = Mock()
            mock_session.begin_transaction.return_value = mock_tx

            from comproscanner.post_processing.visualization.create_knowledge_graph import (
                CreateKG,
            )

            yield CreateKG(), mock_tx

    def _compositions_query_call(self, mock_tx):
        """Find the tx.run call that ran the compositions query"""
        for call in mock_tx.run.call_args_list:
            if "Composition" in call.args[0]:
                return call
        return None

    def test_composition_merge_key_includes_doi(self, mock_kg):
        """The Composition MERGE must key on (composition, doi) so the same
        formula reported by different papers doesn't collide into one node"""
        kg, mock_tx = mock_kg
        synthesis_data = {
            "method": "sol-gel",
            "steps": [],
            "precursors": [],
            "characterization_techniques": [],
        }
        composition_data = {
            "family": "Perovskite",
            "compositions_property_values": {"BaTiO3": 180.0},
            "property_unit": "pC/N",
        }
        paper_metadata = {
            "doi": "10.1000/test1",
            "title": "Test Paper",
            "journal": "Test Journal",
            "year": "2026",
            "isOpenAccess": True,
            "authors": [],
            "keywords": [],
        }

        kg.create_paper_with_compositions(synthesis_data, composition_data, paper_metadata)

        call = self._compositions_query_call(mock_tx)
        assert call is not None
        merge_line = next(
            line for line in call.args[0].splitlines() if "MERGE (c:Composition" in line
        )
        assert "composition: comp_name" in merge_line
        assert "doi: $paper_metadata.doi" in merge_line

    def test_composition_query_uses_paper_doi_param(self, mock_kg):
        """The doi the MERGE key relies on must actually reach the query"""
        kg, mock_tx = mock_kg
        synthesis_data = {
            "method": "sol-gel",
            "steps": [],
            "precursors": [],
            "characterization_techniques": [],
        }
        composition_data = {
            "family": "Perovskite",
            "compositions_property_values": {"BaTiO3": 180.0},
            "property_unit": "pC/N",
        }
        paper_metadata = {
            "doi": "10.1000/test2",
            "title": "Test Paper",
            "journal": "Test Journal",
            "year": "2026",
            "isOpenAccess": True,
            "authors": [],
            "keywords": [],
        }

        kg.create_paper_with_compositions(synthesis_data, composition_data, paper_metadata)

        call = self._compositions_query_call(mock_tx)
        assert call is not None
        assert call.kwargs["paper_metadata"]["doi"] == "10.1000/test2"


NESTED_SCHEMA_YAML = """
version: "1.0"
composition_fields:
  - name: compositions_property_values
    type: "dict[str, object]"
    graph_label: Composition
    relationship_type: MENTIONS
    graph_key_property: composition
    value_fields:
      - name: grain
        type: object
        value_fields:
          - {name: unit, type: str}
          - name: measurements
            type: "dict[str, object]"
            graph_key_property: technique
            value_fields:
              - {name: mean, type: "float|None"}
      - {name: family, type: str, graph_role: node_cluster, graph_label: Family, relationship_type: BELONGS_TO}
synthesis_fields:
  - {name: method, type: str, graph_role: node_cluster, graph_label: Method}
  - {name: notes, type: str, graph_role: skip}
"""


class FakeRecord(dict):
    """Neo4j record stand-in supporting both item access and .get"""


class FakeResult:
    def __init__(self, record=None, records=None):
        self._record = record
        self._records = records or []

    def single(self):
        return self._record

    def data(self):
        return self._records


class FakeTransaction:
    """Records every query and hands out sequential element ids"""

    def __init__(self):
        self.calls = []
        self._next_id = 0

    def _new_id(self):
        self._next_id += 1
        return f"id{self._next_id}"

    def run(self, query, **params):
        self.calls.append((query, params))
        if "UNWIND keys($compositions)" in query:
            return FakeResult(
                records=[{"comp_name": key, "id": self._new_id()} for key in params["compositions"]]
            )
        return FakeResult(record=FakeRecord(id=self._new_id()))

    def commit(self):
        pass

    def rollback(self):
        pass

    def closed(self):
        return False


class TestSchemaDrivenGraph:
    """Test graph writes driven by a nested custom schema"""

    @pytest.fixture
    def kg_and_tx(self, tmp_path):
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text(NESTED_SCHEMA_YAML, encoding="utf-8")
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
        }
        with (
            patch.dict(os.environ, env),
            patch(
                "comproscanner.post_processing.visualization.create_knowledge_graph.GraphDatabase"
            ) as mock_graph_db,
        ):
            from comproscanner.post_processing.visualization.create_knowledge_graph import (
                CreateKG,
            )

            tx = FakeTransaction()
            mock_session = Mock()
            mock_session.run.return_value.single.side_effect = [None, Mock()]
            mock_session.begin_transaction.return_value = tx
            mock_driver = Mock()
            mock_driver.session.return_value = MagicMock()
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_graph_db.driver.return_value = mock_driver
            yield CreateKG(schema_config_path=schema_path), tx

    def write(self, kg):
        composition_data = {
            "compositions_property_values": {
                "BaTiO3": {
                    "grain": {"unit": "um", "measurements": {"SEM": {"mean": 3.1}}},
                    "family": "BT",
                },
                "KNbO3": {"family": "KN"},
            }
        }
        synthesis_data = {"method": "sol-gel", "notes": "ignored"}
        paper_metadata = {"doi": "10.1/x", "title": "T", "authors": [], "keywords": []}
        return kg.create_paper_with_compositions(synthesis_data, composition_data, paper_metadata)

    def queries(self, tx, fragment):
        return [(q, p) for q, p in tx.calls if fragment in q]

    def test_nested_values_follow_json_structure(self, kg_and_tx):
        kg, tx = kg_and_tx
        assert self.write(kg) is True
        # Object block: one child node per composition, owned by its parent
        grain = self.queries(tx, "MERGE (n)-[:HAS_GRAIN]->(c:Grain)")
        assert len(grain) == 1
        # Keyed dictionary inside the block: one node per entry
        measurements = self.queries(tx, "MERGE (n)-[:HAS_MEASUREMENTS]->(c:Measurements {technique: $key})")
        assert [p["key"] for _, p in measurements] == ["SEM"]
        # Leaves become properties of their own parent node
        assert {p["value"] for _, p in self.queries(tx, "SET n.mean = $value")} == {3.1}
        assert {p["value"] for _, p in self.queries(tx, "SET n.unit = $value")} == {"um"}

    def test_nested_cluster_links_each_composition(self, kg_and_tx):
        kg, tx = kg_and_tx
        self.write(kg)
        families = self.queries(tx, "MERGE (v:Family {name: value})")
        assert sorted(p["values"][0] for _, p in families) == ["BT", "KN"]
        assert all("MERGE (n)-[:BELONGS_TO]->(v)" in q for q, _ in families)

    def test_synthesis_fields_hang_off_paper_and_skip_is_respected(self, kg_and_tx):
        kg, tx = kg_and_tx
        self.write(kg)
        method = self.queries(tx, "MERGE (v:Method {name: value})")
        assert method and method[0][1]["values"] == ["sol-gel"]
        assert "MERGE (n)-[:USES_METHOD]->(v)" in method[0][0]
        assert not self.queries(tx, "notes")


class TestDefaultSchemaGraphSettings:
    """The default schema reproduces the legacy graph vocabulary"""

    def test_legacy_labels_and_relationships(self):
        from comproscanner.utils.configs.schema_config import DEFAULT_SCHEMA_CONFIG

        fields = {
            f.name: f
            for domain in ("composition", "synthesis")
            for f in DEFAULT_SCHEMA_CONFIG.fields(domain, final_only=True)
        }
        primary = fields["compositions_property_values"]
        assert (primary.graph_label, primary.relationship_type) == ("Composition", "MENTIONS")
        assert (primary.graph_key_property, primary.graph_value_property) == ("composition", "property_value")
        assert fields["family"].graph_edge_sources == {"paper": "BELONGS_TO_FAMILY", "composition": "BELONGS_TO"}
        assert fields["family"].graph_default == "UNKNOWN_FAMILY"
        assert (fields["method"].graph_label, fields["method"].graph_default) == ("Method", "UNKNOWN_METHOD")
        assert (fields["precursors"].graph_label, fields["precursors"].relationship_type) == ("Precursor", "USED_PRECURSOR")
        assert fields["characterization_techniques"].relationship_type == "USED_CHARACTERIZATION_TECHNIQUE"
        assert (fields["steps"].graph_role, fields["steps"].graph_default) == ("joined_node", "No steps available")
