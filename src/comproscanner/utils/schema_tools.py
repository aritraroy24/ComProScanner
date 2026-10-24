"""
schema_tools.py

Builds runtime artefacts from a loaded ExtractionSchemaConfig: the Pydantic output models the crews enforce, and the prompt text that tells the LLM which fields to produce.
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
    # "object": a nested model built from the field's own value_fields, to any depth.
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
            # Numbers and nested objects default to null, so the type must accept it.
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
    """Build a crew's `output_pydantic` model: `{wrapper_key: {<one attribute per field>}}`.

    Field descriptions are deliberately not attached to the Pydantic fields: CrewAI renders the model schema into its conversion prompt, so they would change that prompt. The LLM sees descriptions through the `{schema_fields_description}` placeholder instead.

    Args:
        fields: Schema fields to include, in output order.
        wrapper_key: Name of the single top-level attribute, e.g. "composition_extracted_data".
        model_name: Class name of the outer model (appears in CrewAI's schema prompt).
        inner_model_name: Class name of the per-field model.
        model_doc: Docstring of the outer model (appears in its JSON schema).
        inner_model_doc: Docstring of the per-field model.
    """
    inner_model = _build_fields_model(fields, inner_model_name, inner_model_doc)
    return create_model(
        model_name, __doc__=model_doc, **{wrapper_key: (inner_model, inner_model())}
    )


Stage = Literal["extraction", "format"]


def render_fields_description(
    fields: List[SchemaField], stage: Stage = "extraction", depth: int = 0
) -> str:
    """Render fields as the bullet list substituted for `{schema_fields_description}`.

    The format stage also shows each field's formatting rule and allowed values. A runtime allowed-values list (`allowed_values_ref`) is emitted as a `{<ref>}` placeholder, which CrewAI fills from the format crew's kickoff inputs.
    """
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
    """Example record for `fields`, from each field's `example` (or `variable_example` for the variable variant), falling back to a type-shaped placeholder."""
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
    """Values for the schema placeholders in a crew's agent/task YAML."""
    return {
        "schema_fields_description": render_fields_description(fields, stage),
        "schema_example_json": render_example_json(fields),
        "schema_field_names": ", ".join(f.name for f in fields),
    }


def fill_schema_placeholders(config: Dict[str, Any], values: Dict[str, str]) -> Dict[str, Any]:
    """Return a copy of an agent/task config with the schema placeholders substituted.

    The crew does this itself, before CrewAI interpolates its kickoff inputs, so the flow's kickoff inputs stay unchanged. Other placeholders (e.g. `{extracted_synthesis_data}`) are left for CrewAI.
    """
    filled = dict(config)
    for key, text in filled.items():
        if isinstance(text, str):
            for name, value in values.items():
                text = text.replace("{" + name + "}", value)
            filled[key] = text
    return filled
