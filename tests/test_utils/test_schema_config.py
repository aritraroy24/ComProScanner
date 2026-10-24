"""
test_schema_config.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

import textwrap

import pytest

from comproscanner.utils.configs.schema_config import (
    DEFAULT_SCHEMA_CONFIG,
    ExtractionSchemaConfig,
    SchemaValidationError,
    load_schema_config,
    parse_type,
)


def write_schema(tmp_path, text):
    path = tmp_path / "schema.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


MINIMAL_SCHEMA = """
    version: "1.0"
    composition_fields:
      - name: compositions_property_values
        type: "dict[str, float|None]"
      - name: property_unit
        type: str
    synthesis_fields:
      - name: method
        type: str
    """


def minimal_schema_dict():
    return {
        "version": "1.0",
        "composition_fields": [
            {"name": "compositions_property_values", "type": "dict[str, float|None]"},
            {"name": "property_unit", "type": "str"},
        ],
        "synthesis_fields": [{"name": "method", "type": "str"}],
    }


class TestDefaultSchema:
    def test_reproduces_historical_fields_in_order(self):
        assert DEFAULT_SCHEMA_CONFIG.field_names("composition", final_only=False) == [
            "compositions_property_values",
            "abbreviations",
            "property_unit",
            "family",
        ]
        assert DEFAULT_SCHEMA_CONFIG.field_names("composition") == [
            "compositions_property_values",
            "property_unit",
            "family",
        ]
        assert DEFAULT_SCHEMA_CONFIG.field_names("synthesis") == [
            "method",
            "precursors",
            "steps",
            "characterization_techniques",
        ]

    def test_default_shapes_match_historical_fallbacks(self):
        assert DEFAULT_SCHEMA_CONFIG.default_shape("composition") == {
            "compositions_property_values": {},
            "property_unit": "",
            "family": "",
        }
        assert DEFAULT_SCHEMA_CONFIG.default_shape("synthesis") == {
            "method": "",
            "precursors": [],
            "steps": [],
            "characterization_techniques": [],
        }

    def test_primary_composition_field(self):
        assert (
            DEFAULT_SCHEMA_CONFIG.primary_composition_field
            == "compositions_property_values"
        )

    def test_default_shape_returns_fresh_copies(self):
        shape = DEFAULT_SCHEMA_CONFIG.default_shape("synthesis")
        shape["precursors"].append("BaCO3")
        assert DEFAULT_SCHEMA_CONFIG.default_shape("synthesis")["precursors"] == []

    def test_load_without_path_returns_default(self):
        assert load_schema_config() is DEFAULT_SCHEMA_CONFIG


class TestParseType:
    @pytest.mark.parametrize(
        "type_str, kind",
        [
            ("str", "str"),
            ("float|int|None", "union"),
            ("list[str]", "list"),
            ("dict[str, int|float|None]", "dict"),
            ("dict[str, object]", "dict"),
            ("object", "object"),
        ],
    )
    def test_supported_types(self, type_str, kind):
        assert parse_type(type_str).kind == kind

    @pytest.mark.parametrize("type_str", ["tuple[int]", "dict[int, str]", "datetime"])
    def test_unsupported_types(self, type_str):
        with pytest.raises(SchemaValidationError):
            parse_type(type_str)


class TestCustomSchema:
    def test_custom_schema_replaces_default_fields(self, tmp_path):
        schema = load_schema_config(write_schema(tmp_path, MINIMAL_SCHEMA))
        assert schema.field_names("composition") == [
            "compositions_property_values",
            "property_unit",
        ]
        assert schema.field_names("synthesis") == ["method"]

    def test_abbreviations_attached_when_not_declared(self, tmp_path):
        schema = load_schema_config(write_schema(tmp_path, MINIMAL_SCHEMA))
        assert schema.field_names("composition", final_only=False) == [
            "compositions_property_values",
            "abbreviations",
            "property_unit",
        ]
        abbreviations = schema.composition_fields[1]
        assert abbreviations.extraction_only
        assert "abbreviation" in abbreviations.description

    def test_own_abbreviations_declaration_is_kept(self, tmp_path):
        text = MINIMAL_SCHEMA.replace(
            "      - name: property_unit\n",
            '      - name: abbreviations\n        type: "dict[str, str]"\n        description: custom\n      - name: property_unit\n',
        )
        schema = load_schema_config(write_schema(tmp_path, text))
        abbreviations = [
            f for f in schema.composition_fields if f.name == "abbreviations"
        ]
        assert len(abbreviations) == 1
        assert abbreviations[0].description == "custom"
        assert not abbreviations[0].extraction_only

    def test_nested_value_fields(self, tmp_path):
        text = """
            version: "1.0"
            composition_fields:
              - name: compositions_property_values
                type: "dict[str, object]"
                value_fields:
                  - name: d33_data
                    type: object
                    value_fields:
                      - {name: value, type: "float|int|None"}
                      - {name: unit, type: str}
                  - {name: sintering_temp, type: "float|None", required: false}
            synthesis_fields:
              - {name: method, type: str}
            """
        schema = load_schema_config(write_schema(tmp_path, text))
        primary = schema.composition_fields[0]
        assert primary.is_object_dict
        assert [f.name for f in primary.value_fields] == ["d33_data", "sintering_temp"]
        assert primary.value_fields[0].is_object
        assert primary.value_fields[0].default is None
        assert primary.value_fields[1].required is False

    def test_derived_defaults(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"] += [
            {"name": "steps", "type": "list[str]"},
            {"name": "temperature", "type": "float"},
            {"name": "is_doped", "type": "bool"},
            {"name": "atmosphere", "type": "str|None"},
        ]
        schema = ExtractionSchemaConfig.from_dict(raw)
        assert schema.default_shape("synthesis") == {
            "method": "",
            "steps": [],
            "temperature": None,
            "is_doped": False,
            "atmosphere": None,
        }


class TestSchemaValidation:
    def assert_rejected(self, raw, message):
        with pytest.raises(SchemaValidationError, match=message):
            ExtractionSchemaConfig.from_dict(raw)

    def test_unsupported_version(self):
        raw = minimal_schema_dict()
        raw["version"] = "2.0"
        self.assert_rejected(raw, "Unsupported schema version")

    @pytest.mark.parametrize("name", ["Method", "method; DROP", "1method", "my-field"])
    def test_unsafe_field_names(self, name):
        raw = minimal_schema_dict()
        raw["synthesis_fields"][0]["name"] = name
        self.assert_rejected(raw, "invalid name")

    def test_unknown_top_level_key(self):
        raw = minimal_schema_dict()
        raw["bogus"] = 1
        self.assert_rejected(raw, "Unknown top-level keys")

    def test_unknown_field_key(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"][0]["colour"] = "red"
        self.assert_rejected(raw, "unknown keys")

    def test_object_without_value_fields(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"][0]["type"] = "object"
        self.assert_rejected(raw, "no value_fields")

    def test_value_fields_on_scalar(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"][0]["value_fields"] = [{"name": "x", "type": "str"}]
        self.assert_rejected(raw, "require type 'object'")

    def test_duplicate_field_names(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"].append({"name": "method", "type": "str"})
        self.assert_rejected(raw, "more than once")

    def test_primary_field_must_be_dict(self):
        raw = minimal_schema_dict()
        raw["composition_fields"][0]["type"] = "str"
        self.assert_rejected(raw, "primary_composition_field")

    def test_synthesis_description_cannot_use_placeholders(self):
        raw = minimal_schema_dict()
        raw["synthesis_fields"][0]["description"] = "Method for {main_extraction_keyword}"
        self.assert_rejected(raw, "unsupported placeholder")

    def test_composition_description_may_use_keyword(self):
        raw = minimal_schema_dict()
        raw["composition_fields"][1]["description"] = "Unit of {main_extraction_keyword}"
        ExtractionSchemaConfig.from_dict(raw)

    def test_unknown_placeholder_rejected(self):
        raw = minimal_schema_dict()
        raw["composition_fields"][1]["format_note"] = "Use {unit_list}"
        self.assert_rejected(raw, "unsupported placeholder")

    @pytest.mark.parametrize(
        "domain, ref",
        [("synthesis_fields", "allowed_atmospheres"), ("composition_fields", "allowed_synthesis_methods")],
    )
    def test_allowed_values_ref_restricted(self, domain, ref):
        raw = minimal_schema_dict()
        raw[domain][-1]["allowed_values_ref"] = ref
        self.assert_rejected(raw, "allowed_values_ref")

    def test_primary_field_must_exist(self):
        raw = minimal_schema_dict()
        raw["primary_composition_field"] = "missing"
        self.assert_rejected(raw, "primary_composition_field")

    def test_missing_file(self, tmp_path):
        with pytest.raises(SchemaValidationError, match="not found"):
            ExtractionSchemaConfig.from_yaml(tmp_path / "missing.yaml")
