"""
schema_config.py

Declarative extraction schema: which fields the composition and synthesis crews extract, their types, defaults and the prompt guidance shown to the LLM. The schema is defined in a YAML file (the bundled `default_extraction_schema.yaml` reproduces ComProScanner's historical output exactly), so a caller can change what gets extracted by pointing `schema_config_path` at their own YAML file without editing any Python source.

A provided schema file replaces the default schema entirely, with one exception: if it does not declare an `abbreviations` field, the default one is attached automatically, since the format crew needs it to expand abbreviated compositions.
"""

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "default_extraction_schema.yaml"
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SCHEMA_DOMAINS = ("composition", "synthesis")

# Field names end up as dict keys, Pydantic attribute names and (later) Cypher property names, so only plain snake_case identifiers are allowed.
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SCALAR_TYPES = {"str", "int", "float", "bool", "None", "object"}
_MISSING = object()


class SchemaValidationError(ValueError):
    """Raised when an extraction schema file is malformed."""


@dataclass
class ParsedType:
    """Structured form of a schema type string such as "dict[str, int|float|None]".

    `kind` is one of the scalar names in `_SCALAR_TYPES`, "list", "dict" or "union"; `args` holds the element type (list), value type (dict) or members (union).
    """

    kind: str
    args: List["ParsedType"] = field(default_factory=list)

    def contains(self, kind: str) -> bool:
        return self.kind == kind or any(arg.contains(kind) for arg in self.args)

    def is_nullable(self) -> bool:
        return self.kind == "None" or (
            self.kind == "union" and any(a.kind == "None" for a in self.args)
        )


def _split_top_level(text: str, sep: str) -> List[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def parse_type(type_str: str) -> ParsedType:
    """Parse a schema type string into a `ParsedType`, raising SchemaValidationError if unsupported."""
    text = re.sub(r"\s+", "", str(type_str))
    members = _split_top_level(text, "|")
    if len(members) > 1:
        return ParsedType("union", [parse_type(m) for m in members])
    if text in _SCALAR_TYPES:
        return ParsedType(text)
    match = re.fullmatch(r"(list|dict)\[(.*)\]", text)
    if match:
        container, inner = match.groups()
        if container == "list":
            return ParsedType("list", [parse_type(inner)])
        key_and_value = _split_top_level(inner, ",")
        if len(key_and_value) != 2 or key_and_value[0] != "str":
            raise SchemaValidationError(
                f"Unsupported dict type '{type_str}': only dict[str, <value type>] is supported"
            )
        return ParsedType("dict", [parse_type(key_and_value[1])])
    raise SchemaValidationError(f"Unsupported field type '{type_str}'")


@dataclass
class SchemaField:
    """One extracted field.

    Attributes:
        name: Dict key used in the extracted record.
        type: Type string, e.g. "str", "list[str]", "dict[str, int|float|None]", "object", "dict[str, object]".
        description: Guidance shown to the LLM. "{main_extraction_keyword}" is substituted at prompt-build time.
        required: Prompt-level only (rendered as required/optional). Every generated Pydantic field keeps a default, so a missing key never fails validation.
        default: Value used when the field is absent. Derived from the type when not set.
        allowed_values: Inline list of allowed values the formatter maps this field onto.
        allowed_values_ref: Name of a runtime-supplied allowed-values list (e.g. "allowed_synthesis_methods").
        value_fields: Sub-fields for "object" / "dict[str, object]" types.
        extraction_only: Field exists only in the extraction crew's output and is consumed (dropped) by the format crew, like `abbreviations`.
        format_note: Field-specific rule shown to the format crew.
        example: Example value used to render `{schema_example_json}`.
        variable_example: Example value for the variable-composition example; falls back to `example`.
    """

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = _MISSING
    allowed_values: Optional[List[str]] = None
    allowed_values_ref: Optional[str] = None
    value_fields: Optional[List["SchemaField"]] = None
    extraction_only: bool = False
    format_note: Optional[str] = None
    example: Any = None
    variable_example: Any = None
    parsed_type: ParsedType = field(init=False, repr=False)

    def __post_init__(self):
        self.parsed_type = parse_type(self.type)
        if self.default is _MISSING:
            self.default = self._derive_default()

    @property
    def is_object(self) -> bool:
        return self.parsed_type.kind == "object"

    @property
    def is_object_dict(self) -> bool:
        return self.parsed_type.kind == "dict" and self.parsed_type.args[0].contains(
            "object"
        )

    def _derive_default(self) -> Any:
        kind = self.parsed_type.kind
        if self.parsed_type.is_nullable():
            return None
        if kind == "str":
            return ""
        if kind == "list":
            return []
        if kind == "dict":
            return {}
        if kind == "bool":
            return False
        # Numbers and nested objects have no natural empty value, so they default to null.
        return None

    def get_default(self) -> Any:
        """Return a fresh copy of the default so callers can't mutate the shared value."""
        return copy.deepcopy(self.default)


_FIELD_KEYS = {
    "name",
    "type",
    "description",
    "required",
    "default",
    "allowed_values",
    "allowed_values_ref",
    "value_fields",
    "extraction_only",
    "format_note",
    "example",
    "variable_example",
}


def _parse_fields(raw_fields: Any, location: str) -> List[SchemaField]:
    if not isinstance(raw_fields, list) or not raw_fields:
        raise SchemaValidationError(f"{location} must be a non-empty list of fields")
    fields, seen = [], set()
    for index, raw in enumerate(raw_fields):
        where = f"{location}[{index}]"
        if not isinstance(raw, dict):
            raise SchemaValidationError(f"{where} must be a mapping")
        unknown = set(raw) - _FIELD_KEYS
        if unknown:
            raise SchemaValidationError(f"{where} has unknown keys: {sorted(unknown)}")
        for key in ("name", "type"):
            if key not in raw:
                raise SchemaValidationError(f"{where} is missing '{key}'")
        name = raw["name"]
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            raise SchemaValidationError(
                f"{where} has invalid name {name!r}: must match {_SAFE_NAME.pattern}"
            )
        if name in seen:
            raise SchemaValidationError(f"{location} declares '{name}' more than once")
        seen.add(name)

        kwargs = {k: v for k, v in raw.items() if k != "value_fields"}
        if raw.get("value_fields") is not None:
            kwargs["value_fields"] = _parse_fields(
                raw["value_fields"], f"{where}.value_fields"
            )
        try:
            schema_field = SchemaField(**kwargs)
        except SchemaValidationError as e:
            raise SchemaValidationError(f"{where}: {e}") from None

        needs_value_fields = schema_field.is_object or schema_field.is_object_dict
        if needs_value_fields and not schema_field.value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}') has type '{schema_field.type}' but no value_fields"
            )
        if schema_field.value_fields and not needs_value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}') declares value_fields, which require type 'object' or 'dict[str, object]'"
            )
        if schema_field.parsed_type.contains("object") and not needs_value_fields:
            raise SchemaValidationError(
                f"{where} ('{name}'): 'object' is only supported as 'object' or 'dict[str, object]'"
            )
        fields.append(schema_field)
    return fields


BUILTIN_ABBREVIATIONS_FIELD = "abbreviations"


def _attach_builtin_abbreviations(
    composition_fields: List[SchemaField], primary: Optional[str]
) -> List[SchemaField]:
    """Attach the default schema's `abbreviations` field to a schema that doesn't declare one.

    A custom schema replaces every other field, but abbreviations are always kept: the format crew needs them to expand abbreviated compositions into full formulas, and without them those compositions would be lost in cleaning. A schema that declares its own `abbreviations` field keeps its own declaration. The attached field goes right after the primary composition field, matching the default schema's order.
    """
    if any(f.name == BUILTIN_ABBREVIATIONS_FIELD for f in composition_fields):
        return composition_fields
    with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        default_raw = yaml.safe_load(f)
    builtin = next(
        f
        for f in _parse_fields(
            default_raw["composition_fields"], "default composition_fields"
        )
        if f.name == BUILTIN_ABBREVIATIONS_FIELD
    )
    names = [f.name for f in composition_fields]
    if primary not in names:
        primary = next(
            (f.name for f in composition_fields if f.parsed_type.kind == "dict"), None
        )
    insert_at = names.index(primary) + 1 if primary in names else len(names)
    return composition_fields[:insert_at] + [builtin] + composition_fields[insert_at:]


@dataclass
class ExtractionSchemaConfig:
    """The loaded extraction schema.

    Attributes:
        version: Schema format version.
        composition_fields: Fields of the `composition_data` section.
        synthesis_fields: Fields of the `synthesis_data` section.
        primary_composition_field: The composition-keyed field that decides whether a paper has usable data and that the cleaning steps operate on.
        extra_domains: Further field groups, parsed and validated but not yet extracted (the output envelope stays fixed for now).
        source_path: File the schema was loaded from.
    """

    version: str
    composition_fields: List[SchemaField]
    synthesis_fields: List[SchemaField]
    primary_composition_field: str
    extra_domains: Dict[str, List[SchemaField]] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExtractionSchemaConfig":
        """Load and validate a schema file. The file fully replaces the default schema."""
        path = Path(path)
        if not path.is_file():
            raise SchemaValidationError(f"Schema file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        try:
            return cls.from_dict(raw, source_path=path)
        except SchemaValidationError as e:
            raise SchemaValidationError(f"{path}: {e}") from None

    @classmethod
    def from_dict(
        cls, raw: Any, source_path: Optional[Path] = None
    ) -> "ExtractionSchemaConfig":
        if not isinstance(raw, dict):
            raise SchemaValidationError("Schema must be a YAML mapping")
        version = str(raw.get("version", ""))
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaValidationError(
                f"Unsupported schema version {version!r}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        unknown = set(raw) - {
            "version",
            "composition_fields",
            "synthesis_fields",
            "primary_composition_field",
            "extra_domains",
        }
        if unknown:
            raise SchemaValidationError(f"Unknown top-level keys: {sorted(unknown)}")

        composition_fields = _parse_fields(
            raw.get("composition_fields"), "composition_fields"
        )
        composition_fields = _attach_builtin_abbreviations(
            composition_fields, raw.get("primary_composition_field")
        )
        synthesis_fields = _parse_fields(raw.get("synthesis_fields"), "synthesis_fields")
        extra_domains = {}
        for domain, domain_fields in (raw.get("extra_domains") or {}).items():
            if not _SAFE_NAME.match(str(domain)) or domain in SCHEMA_DOMAINS:
                raise SchemaValidationError(f"Invalid extra domain name {domain!r}")
            extra_domains[domain] = _parse_fields(
                domain_fields, f"extra_domains.{domain}"
            )

        primary = raw.get("primary_composition_field")
        if primary is None:
            primary = next(
                (f.name for f in composition_fields if f.parsed_type.kind == "dict"),
                None,
            )
        primary_field = next(
            (f for f in composition_fields if f.name == primary), None
        )
        if primary_field is None:
            raise SchemaValidationError(
                f"primary_composition_field {primary!r} is not declared in composition_fields"
            )
        if primary_field.parsed_type.kind != "dict" or primary_field.extraction_only:
            raise SchemaValidationError(
                f"primary_composition_field '{primary}' must be a dict-typed field kept in the final output"
            )

        return cls(
            version=version,
            composition_fields=composition_fields,
            synthesis_fields=synthesis_fields,
            primary_composition_field=primary,
            extra_domains=extra_domains,
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "ExtractionSchemaConfig":
        """Load `path`, or the bundled default schema when `path` is None."""
        return cls.from_yaml(path if path is not None else DEFAULT_SCHEMA_PATH)

    def fields(self, domain: str, final_only: bool = False) -> List[SchemaField]:
        """Fields of `domain` ("composition" or "synthesis"); `final_only` drops extraction-only fields."""
        if domain == "composition":
            domain_fields = self.composition_fields
        elif domain == "synthesis":
            domain_fields = self.synthesis_fields
        else:
            raise ValueError(f"Unknown schema domain '{domain}'")
        if final_only:
            return [f for f in domain_fields if not f.extraction_only]
        return list(domain_fields)

    def field_names(self, domain: str, final_only: bool = True) -> List[str]:
        return [f.name for f in self.fields(domain, final_only=final_only)]

    def default_shape(self, domain: str, final_only: bool = True) -> Dict[str, Any]:
        """Empty record for `domain`, used wherever a fallback or placeholder record is needed."""
        return {f.name: f.get_default() for f in self.fields(domain, final_only)}


DEFAULT_SCHEMA_CONFIG = ExtractionSchemaConfig.load()


def load_schema_config(
    schema_config_path: Optional[Union[str, Path]] = None,
) -> ExtractionSchemaConfig:
    """Return the schema at `schema_config_path`, or the shared default when it is None."""
    if schema_config_path is None:
        return DEFAULT_SCHEMA_CONFIG
    return ExtractionSchemaConfig.from_yaml(schema_config_path)
