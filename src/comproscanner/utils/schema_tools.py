"""
schema_tools.py

Builds runtime artefacts from a loaded ExtractionSchemaConfig: the Pydantic output models the crews enforce.
"""

from typing import Any, Dict, List, Optional, Type, Union

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
