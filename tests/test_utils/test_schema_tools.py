"""
test_schema_tools.py

Regression tests for the schema-generated crew output models. The fixture snapshot was captured from the hand-written Pydantic models these replace, before they were removed, so matching it proves the default schema leaves the crews' enforced output shape (and the schema text CrewAI shows the LLM, which is built from these annotations) unchanged.
"""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from comproscanner.utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    ExtractionSchemaConfig,
)
from comproscanner.utils.schema_tools import build_output_model

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_output_models_snapshot.json").read_text(
        encoding="utf-8"
    )
)

# Arguments each crew passes to build_output_model.
CREW_MODELS = {
    "composition_extraction": dict(
        domain="composition",
        final_only=False,
        wrapper_key="composition_extracted_data",
        model_name="ExtractedCompositionPropertyData",
        inner_model_name="DetailedCompositionPropertyData",
        model_doc="Extracted Composition Property",
        inner_model_doc="Detailed Composition Property",
    ),
    "composition_format": dict(
        domain="composition",
        final_only=True,
        wrapper_key="composition_formatted_data",
        model_name="FormattedCompositionPropertyData",
        inner_model_name="DetailedCompositionPropertyData",
        model_doc="Formatted Composition Property",
        inner_model_doc="Detailed Composition Property",
    ),
    "synthesis_extraction": dict(
        domain="synthesis",
        final_only=False,
        wrapper_key="synthesis_extracted_data",
        model_name="ExtractedSynthesisData",
        inner_model_name="DetailedSynthesisData",
        model_doc="Extracted Synthesis Data",
        inner_model_doc="Detailed Synthesis Data",
    ),
    "synthesis_format": dict(
        domain="synthesis",
        final_only=True,
        wrapper_key="synthesis_formatted_data",
        model_name="FormattedSynthesisData",
        inner_model_name="DetailedSynthesisData",
        model_doc="Formatted Synthesis Data",
        inner_model_doc="Detailed Synthesis Data",
    ),
}


def build(schema, domain, final_only, **kwargs):
    return build_output_model(schema.fields(domain, final_only=final_only), **kwargs)


def annotations(model):
    return {
        name: annotations(f.annotation)
        if isinstance(f.annotation, type) and issubclass(f.annotation, BaseModel)
        else str(f.annotation)
        for name, f in model.model_fields.items()
    }


@pytest.mark.parametrize("crew", sorted(CREW_MODELS))
class TestDefaultSchemaMatchesLegacyModels:
    def test_json_schema(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert model.model_json_schema() == SNAPSHOT[crew]["model_json_schema"]

    def test_default_dump(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert model().model_dump() == SNAPSHOT[crew]["default_dump"]

    def test_annotations(self, crew):
        model = build(DEFAULT_SCHEMA_CONFIG, **CREW_MODELS[crew])
        assert annotations(model) == SNAPSHOT[crew]["annotations"]


class TestCustomSchemaModels:
    @pytest.fixture
    def branched_schema(self):
        return ExtractionSchemaConfig.from_dict(
            {
                "version": "1.0",
                "composition_fields": [
                    {
                        "name": "compositions_property_values",
                        "type": "dict[str, object]",
                        "value_fields": [
                            {
                                "name": "d33_data",
                                "type": "object",
                                "value_fields": [
                                    {"name": "value", "type": "float|int|None"},
                                    {"name": "unit", "type": "str"},
                                ],
                            },
                            {"name": "sintering_temp", "type": "float|None"},
                        ],
                    },
                    {"name": "property_unit", "type": "str"},
                ],
                "synthesis_fields": [
                    {"name": "method", "type": "str"},
                    {"name": "sintering_atmosphere", "type": "str", "required": False},
                ],
            }
        )

    def test_branched_composition_values_validate(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["composition_format"])
        record = {
            "composition_formatted_data": {
                "compositions_property_values": {
                    "BaTiO3": {
                        "d33_data": {"value": 190, "unit": "pC/N"},
                        "sintering_temp": 1300,
                    },
                    "Ba0.9Ca0.1TiO3": {"d33_data": {"value": 250, "unit": "pC/N"}},
                },
                "property_unit": "pC/N",
            }
        }
        dumped = model.model_validate(record).model_dump()
        values = dumped["composition_formatted_data"]["compositions_property_values"]
        assert values["BaTiO3"]["d33_data"] == {"value": 190, "unit": "pC/N"}
        assert values["BaTiO3"]["sintering_temp"] == 1300
        # An omitted optional nested block falls back to its default instead of failing validation.
        assert values["Ba0.9Ca0.1TiO3"]["sintering_temp"] is None

    def test_abbreviations_kept_for_extraction_only(self, branched_schema):
        extraction = build(branched_schema, **CREW_MODELS["composition_extraction"])
        formatted = build(branched_schema, **CREW_MODELS["composition_format"])
        assert "abbreviations" in extraction().composition_extracted_data.model_dump()
        assert "abbreviations" not in formatted().composition_formatted_data.model_dump()

    def test_extra_synthesis_field(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["synthesis_format"])
        assert model().model_dump() == {
            "synthesis_formatted_data": {"method": "", "sintering_atmosphere": ""}
        }

    def test_nested_objects_accept_null(self, branched_schema):
        model = build(branched_schema, **CREW_MODELS["composition_format"])
        record = {
            "composition_formatted_data": {
                "compositions_property_values": {"BaTiO3": {"d33_data": None}}
            }
        }
        dumped = model.model_validate(record).model_dump()
        assert (
            dumped["composition_formatted_data"]["compositions_property_values"][
                "BaTiO3"
            ]["d33_data"]
            is None
        )
