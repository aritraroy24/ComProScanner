# Dynamic extraction schema — Phase 1

Branch: `feat/dynamic-extraction-schema`. Source plan: `DYNAMIC_SCHEMA_PLAN.md`. Check in with the user after Phase 1 before starting Phase 2.

## Phase 1 checklist

- [x] 1. Audit `post_processing/data_cleaner.py` for hardcoded field names (findings below)
- [x] 2. `utils/configs/default_extraction_schema.yaml` + `utils/configs/schema_config.py` (`SchemaField`, `ExtractionSchemaConfig.from_yaml()`, `DEFAULT_SCHEMA_CONFIG`)
- [x] 3. `build_output_model()` + wire into the 4 crew files, replacing the hand-written Pydantic classes
- [x] 4. Thread `schema_config_path` through `ComProScanner(...)`, `DataExtractionFlow.__init__` and `DataCleaner`
- [x] 5. Replace literal `expected_keys` / `composition_expected_keys` / duplicated `default_synthesis_data` in `main_extraction_flow.py`
- [x] 6. `{schema_fields_description}` / `{schema_example_json}` placeholders; rewrite the 8 crew YAML files; fix `'methods'` → `method`
- [x] 7. Regression tests (see "Regression strategy")

## Step 1 audit: `data_cleaner.py`

- `_get_comp_prop_data` (L74), `clean_data_with_selected_steps` (L1748), `get_all_composition_property_pairs` (L1819, L1842) hardcode `compositions_property_values` → use `schema_config.primary_composition_field`.
- `get_useful_data` (L1624-1679) hardcodes the full default record shape (both domains, all fields minus `abbreviations`) → derive from `schema_config.default_shape(domain)`; `article_metadata` stays fixed (envelope scope boundary).
- Cleaning steps only rewrite composition *keys*; values pass through untouched, so branched `value_fields` values survive cleaning unchanged. `get_all_composition_property_pairs` filters `prop is not None`, which also works for object values.

## Design decisions (not spelled out in the plan)

- **`extraction_only: true`** on `SchemaField`: today the composition *format* model and every post-format consumer omit `abbreviations` (the formatter consumes it to expand compositions). The flag keeps that split declarative instead of special-casing the name.
- **`abbreviations` is always attached to a custom schema** (user decision, 2026-10-24). A custom schema replaces everything else, but if it doesn't declare `abbreviations`, the default schema's field is inserted right after the primary composition field. Without it the formatter couldn't expand abbreviated compositions and those compositions would be lost in cleaning. A schema that declares its own `abbreviations` keeps its own.
- **`example` / `variable_example`** on `SchemaField`: `{schema_example_json}` needs example values. The bundled YAML carries today's BaTiO3 / NaNbO3 examples. User-passed `expected_*_example` strings still override (public API unchanged).
- **`required` is prompt-level only.** Every generated Pydantic field keeps a default, exactly like the current models (none are required at the Pydantic level today), so a missing key never turns into a validation failure.
- **Field descriptions are not passed into Pydantic `Field(description=...)`**, because CrewAI renders the model schema into the conversion prompt and that would change it. Descriptions reach the LLM through `{schema_fields_description}` instead.
- **Model/class names and docstrings are preserved** (`ExtractedCompositionPropertyData` etc.) since they appear in CrewAI's schema prompt.
- Field-specific prompt guidance (abbreviation rules, "empty list if precursors not mentioned", allowed-values rules) moves into each field's `description` in the schema YAML, so it disappears with the field. Generic instructions (Equation Tool, Graph tool, notation rules) stay in the crew YAML.
- Safe-identifier validation (`^[a-z][a-z0-9_]*$`) for field names is done at load time now rather than in Phase 3, since it costs nothing.

## Regression strategy

Byte-for-byte output diffs on live DOIs aren't meaningful: LLM output is non-deterministic run to run, and step 6 deliberately changes prompt text. Deterministic substitutes:

1. Snapshot of the 4 legacy models' `model_json_schema()`, CrewAI `generate_model_description()`, `PydanticSchemaParser.get_schema()` and default dump (`tests/test_utils/fixtures/legacy_output_models_snapshot.json`, captured before any change); generated models must match exactly.
2. Flow-level tests with stubbed crews: identical envelopes, defaults and parsing for the default schema.
3. Existing `tests/` pass unmodified.
4. A custom-schema test (extra synthesis field + branched `value_fields`) proving no `.py` edit is needed.

## Review

Phase 1 is complete in commits 7f292f3 (schema config), 06c8edb (generated models), 87d1cda (schema threaded through ComProScanner/flow/DataCleaner), d689b61 (schema-driven prompts).

Verified:
- The four generated output models are identical to the legacy hand-written ones: `model_json_schema()`, default dump, CrewAI `generate_model_description()` and `PydanticSchemaParser.get_schema()` (checked with real CrewAI outside pytest; the first two and the annotations are also pytest regression tests against the snapshot).
- All four crews' prompts pass real CrewAI interpolation with the flow's exact kickoff inputs, for both the default schema and a branched custom schema, with no leftover placeholders.
- Full suite: 653 passed, 1 failed (`test_embeddings.py::test_embed_documents_huggingface`, already failing on `main` before this branch). Existing tests pass unmodified.

Deliberate behaviour changes with the default schema:
- Prompt text changed (field list + rendered examples instead of hand-written prose). The LLM now sees the correct `compositions_property_values` key in the examples instead of `compositions`.
- The synthesis "nothing to format" check looks at every synthesis field instead of method/precursors/steps only. With the default schema this path is only reached with all fields empty, so it's equivalent in practice.

Not done: a live LLM run on fixed DOIs. Byte-identical output isn't achievable (LLM output varies run to run, and the prompts deliberately changed), so the deterministic checks above stand in for it. A small live spot-check before merging is still worthwhile.
