"""
schema_tools.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

import json
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, create_model

from .configs.schema_config import ParsedType, SchemaField

_ATOMIC_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "None": type(None)}


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _annotation(parsed: ParsedType, schema_field: SchemaField, owner: str) -> Any:
    if parsed.kind in _ATOMIC_TYPES:
        return _ATOMIC_TYPES[parsed.kind]
    if parsed.kind == "union":
        members = tuple(_annotation(a, schema_field, owner) for a in parsed.args)
        return Union[members]
    if parsed.kind == "list":
        return List[_annotation(parsed.args[0], schema_field, owner)]
    if parsed.kind == "dict":
        return Dict[str, _annotation(parsed.args[0], schema_field, owner)]
    # Nested model from value_fields
    return _build_fields_model(
        schema_field.value_fields, f"{owner}{_camel(schema_field.name)}"
    )


def _build_fields_model(
    fields: List[SchemaField], model_name: str, doc: Optional[str] = None
) -> Type[BaseModel]:
    definitions = {}
    for schema_field in fields:
        annotation = _annotation(schema_field.parsed_type, schema_field, model_name)
        default = schema_field.get_default()
        if default is None and not schema_field.parsed_type.is_nullable():
            # Allow null default
            annotation = Optional[annotation]
        definitions[schema_field.name] = (annotation, default)
    return create_model(model_name, __doc__=doc, **definitions)


def build_output_model(
    fields: List[SchemaField],
    wrapper_key: str,
    model_name: str,
    inner_model_name: str,
    model_doc: Optional[str] = None,
    inner_model_doc: Optional[str] = None,
) -> Type[BaseModel]:
    """
    Build a crew's output_pydantic model with a single wrapper key around the schema fields.

    Args:
        fields (list): Schema fields, in output order.
        wrapper_key (str): Top-level key, e.g. "composition_extracted_data".
        model_name (str): Class name of the outer model.
        inner_model_name (str): Class name of the inner model.
        model_doc (str, optional): Docstring of the outer model.
        inner_model_doc (str, optional): Docstring of the inner model.
    """
    inner_model = _build_fields_model(fields, inner_model_name, inner_model_doc)
    return create_model(
        model_name, __doc__=model_doc, **{wrapper_key: (inner_model, inner_model())}
    )


Stage = Literal["extraction", "format"]


def render_fields_description(
    fields: List[SchemaField], stage: Stage = "extraction", depth: int = 0
) -> str:
    """Render fields as a bullet list for the schema_fields_description placeholder."""
    pad = "  " * depth
    lines = []
    for schema_field in fields:
        requirement = "required" if schema_field.required else "optional"
        line = f"{pad}- {schema_field.name} ({schema_field.type}, {requirement})"
        if schema_field.description:
            line += f": {schema_field.description}"
        lines.append(line)
        if stage == "format":
            if schema_field.format_note:
                lines.append(f"{pad}  Formatting rule: {schema_field.format_note}")
            if schema_field.allowed_values:
                lines.append(
                    f"{pad}  Allowed values: {', '.join(schema_field.allowed_values)}"
                )
            elif schema_field.allowed_values_ref:
                lines.append(
                    f"{pad}  Allowed values:\n{{{schema_field.allowed_values_ref}}}"
                )
        if schema_field.value_fields:
            if schema_field.is_object_dict:
                lines.append(f"{pad}  Each value is an object with these fields:")
            else:
                lines.append(f"{pad}  An object with these fields:")
            lines.append(
                render_fields_description(schema_field.value_fields, stage, depth + 2)
            )
    return "\n".join(lines)


def _placeholder_value(parsed: ParsedType, schema_field: SchemaField, variant: str) -> Any:
    kind = parsed.kind
    if kind == "union":
        members = [a for a in parsed.args if a.kind != "None"] or parsed.args
        return _placeholder_value(members[0], schema_field, variant)
    if kind == "str":
        return f"<{schema_field.name}>"
    if kind in ("int", "float"):
        return 0
    if kind == "bool":
        return False
    if kind == "None":
        return None
    if kind == "list":
        return [_placeholder_value(parsed.args[0], schema_field, variant)]
    if kind == "dict":
        return {"<key>": _placeholder_value(parsed.args[0], schema_field, variant)}
    return build_example(schema_field.value_fields, variant)


def build_example(fields: List[SchemaField], variant: str = "default") -> Dict[str, Any]:
    """Build an example record from the fields' example values."""
    example = {}
    for schema_field in fields:
        value = schema_field.example
        if variant == "variable" and schema_field.variable_example is not None:
            value = schema_field.variable_example
        if value is None:
            value = _placeholder_value(schema_field.parsed_type, schema_field, variant)
        example[schema_field.name] = value
    return example


def render_example_json(fields: List[SchemaField], variant: str = "default") -> str:
    return json.dumps(build_example(fields, variant), indent=2, ensure_ascii=False)


def schema_prompt_values(fields: List[SchemaField], stage: Stage) -> Dict[str, str]:
    """Values for the schema placeholders in agent/task YAML."""
    return {
        "schema_fields_description": render_fields_description(fields, stage),
        "schema_example_json": render_example_json(fields),
        "schema_field_names": ", ".join(f.name for f in fields),
    }


def fill_schema_placeholders(config: Dict[str, Any], values: Dict[str, str]) -> Dict[str, Any]:
    """Return a copy of an agent/task config with the schema placeholders filled in."""
    filled = dict(config)
    for key, text in filled.items():
        if isinstance(text, str):
            for name, value in values.items():
                text = text.replace("{" + name + "}", value)
            filled[key] = text
    return filled
