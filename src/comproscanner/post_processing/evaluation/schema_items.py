"""
schema_items.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
GitHub: aritraroy24
"""

# Standard library imports
from typing import Any, Dict, Iterator, List, Tuple

# Custom imports
from ...utils.configs.schema_config import ExtractionSchemaConfig, SchemaField


def leaf_values(
    fields: List[SchemaField], value: Any, path: Tuple[str, ...] = ()
) -> Iterator[Tuple[SchemaField, Tuple[str, ...], Any]]:
    """Yield (leaf_field, path, value) for every leaf of a nested value.

    Args:
        fields (List[SchemaField], required): Schema fields to walk, at the current nesting level.
        value (Any, required): The data value corresponding to `fields`, typically a dict.
        path (Tuple[str, ...], optional): Dotted-path prefix accumulated from parent calls. Defaults to ().

    Returns:
        Iterator[Tuple[SchemaField, Tuple[str, ...], Any]]: One tuple per leaf field, giving the leaf's
        schema field, its full path (including the entry key for nested keyed dictionaries), and its value.
    """
    for field in fields:
        sub_value = value.get(field.name) if isinstance(value, dict) else None
        if field.value_fields and field.is_object:
            yield from leaf_values(field.value_fields, sub_value, path + (field.name,))
        elif field.value_fields and isinstance(sub_value, dict):
            # Nested keyed dictionary: each entry's leaves are separate items
            for key in sorted(sub_value):
                yield from leaf_values(
                    field.value_fields, sub_value[key], path + (field.name, key)
                )
        else:
            yield field, path + (field.name,), sub_value


def leaf_weights(field: SchemaField) -> Dict[str, float]:
    """Effective weight of every leaf of a field with value_fields, keyed by dotted path.

    Args:
        field (SchemaField, required): Field whose nested value_fields are walked for leaf weights.

    Returns:
        Dict[str, float]: Mapping from dotted leaf path to that leaf's effective weight.
    """
    return {
        ".".join(path): leaf.effective_weight
        for leaf, path, _ in leaf_values(field.value_fields, None)
    }


def count_value_items(field: SchemaField, value: Any) -> int:
    """Number of evaluation items in one value of a field (one per present leaf).

    Args:
        field (SchemaField, required): Field definition, used to find its nested value_fields if any.
        value (Any, required): The value to count leaves within.

    Returns:
        int: 1 if the field has no value_fields, otherwise the count of present leaves.
    """
    if not field.value_fields:
        return 1
    return sum(
        1 for _, _, leaf_value in leaf_values(field.value_fields, value) if leaf_value is not None
    )


def count_record_items(schema_config: ExtractionSchemaConfig, item: Dict[str, Any]) -> int:
    """Count all individual evaluation items in a data entry.

    Args:
        schema_config (ExtractionSchemaConfig, required): Schema config providing the fields for each domain.
        item (Dict[str, Any], required): Data entry containing composition_data and/or synthesis_data sections.

    Returns:
        int: Total number of individual evaluation items across both domains.
    """
    count = 0
    for domain in ("composition", "synthesis"):
        section = item.get(f"{domain}_data", {})
        if section is None:
            continue
        for field in schema_config.fields(domain, final_only=True):
            if field.match_mode == "keyed_values":
                # Keys and value items
                count += sum(
                    1 + count_value_items(field, value)
                    for value in (section.get(field.name) or {}).values()
                )
            elif field.match_mode == "item_list":
                count += len(section.get(field.name) or [])
            elif field.match_mode == "joined_text":
                if section.get(field.name):
                    count += 2  # Count as 2 items (presence + content)
            elif field.match_mode is None:
                count += count_value_items(field, section.get(field.name))
            elif field.name in section:
                count += 1
    return count
