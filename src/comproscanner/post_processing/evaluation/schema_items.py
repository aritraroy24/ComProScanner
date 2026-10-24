"""
schema_items.py

Author: Aritra Roy
Email: contact@aritraroy.live
Website: https://aritraroy.live
Date: 24-10-2026
"""

# Standard library imports
from typing import Any, Dict, Iterator, List, Tuple

# Custom imports
from ...utils.configs.schema_config import ExtractionSchemaConfig, SchemaField


def leaf_values(
    fields: List[SchemaField], value: Any, path: Tuple[str, ...] = ()
) -> Iterator[Tuple[SchemaField, Tuple[str, ...], Any]]:
    """Yield (leaf_field, path, value) for every leaf of a nested value."""
    for field in fields:
        sub_value = value.get(field.name) if isinstance(value, dict) else None
        if field.value_fields and field.is_object:
            yield from leaf_values(field.value_fields, sub_value, path + (field.name,))
        else:
            yield field, path + (field.name,), sub_value


def leaf_weights(field: SchemaField) -> Dict[str, float]:
    """Effective weight of every leaf of a field with value_fields, keyed by dotted path."""
    return {
        ".".join(path): leaf.effective_weight
        for leaf, path, _ in leaf_values(field.value_fields, None)
    }


def count_value_items(field: SchemaField, value: Any) -> int:
    """Number of evaluation items in one value of a field (one per present leaf)."""
    if not field.value_fields:
        return 1
    return sum(
        1 for _, _, leaf_value in leaf_values(field.value_fields, value) if leaf_value is not None
    )


def count_record_items(schema_config: ExtractionSchemaConfig, item: Dict[str, Any]) -> int:
    """Count all individual evaluation items in a data entry."""
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
