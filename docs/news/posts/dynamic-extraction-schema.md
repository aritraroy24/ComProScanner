---
date:
  created: 2026-10-26
authors:
  - aritraroy24
description: >-
  ComProScanner now reads what to extract from a YAML schema. Fields can be
  nested to any depth, and the same file drives extraction, cleaning, both
  evaluators, the plots and the Neo4j knowledge graph, with the default
  schema reproducing every earlier result exactly.
---

# Extract Anything: A Custom Schema for ComProScanner

<!-- more -->

## Overview

Until now, ComProScanner extracted one fixed set of fields: composition–property values with their unit and material family, and the synthesis method, precursors, steps and characterization techniques. That set suited the piezoelectric study the framework was built for<sup>1</sup>, but researchers kept asking for something slightly different: the measurement conditions of a value, grain sizes, crystal phases, sintering profiles. Each of these meant editing Python code in several places.

This update moves the definition of *what is extracted* out of the code and into a single YAML file, the **extraction schema**. A field is declared once, with its type, description and example, and every stage of the pipeline follows it: the extraction prompts and output validation, data cleaning, semantic and agentic evaluation, distribution plots and the Neo4j knowledge graph. Fields can be nested to any depth. The bundled default schema reproduces the earlier behaviour exactly, which was verified against the previous code on every stage. This post explains the motivation, the design decisions and how the change was verified.

---

## 1. Motivation: Seven Keys in Many Places

The seven output keys were spelled out independently in many parts of the code:

- four near-duplicate Pydantic output models for the extraction and formatting agents;
- eight agent and task prompt files, which named the fields in prose;
- the flow, which assembled and validated the output envelope;
- the data cleaner, which looked up `compositions_property_values` by name;
- the semantic evaluator, with one hand-written comparison block per field;
- the agentic evaluator, with its own response models and a prompt that listed every field;
- the distribution plots, with separate functions for families, precursors and techniques;
- the evaluation charts, with four hand-written per-field metrics;
- the knowledge-graph builder, with per-field Cypher queries.

Adding one field meant touching all of these consistently, and they had already drifted apart. The few-shot example in the prompts, for instance, used the key `"compositions"` while the validated output model required `compositions_property_values`.

---

## 2. The Extraction Schema

A schema file lists the fields of the two data sections of a record, `composition_data` and `synthesis_data`. A field needs only a name and a type; everything else is optional:

```yaml
version: "1.0"
primary_composition_field: compositions_property_values

composition_fields:
  - name: compositions_property_values
    type: "dict[str, object]"
    description: Every composition reported in the article's own work, mapped to its measured data.
    value_fields:
      - name: d33_data
        type: object
        value_fields:
          - {name: value, type: "float|int|None", similarity_threshold: 0.05, example: 620}
          - {name: unit, type: str, match_mode: exact, example: pC/N}
      - name: phase
        type: object
        value_fields:
          - {name: crystal_structure, type: str, graph_role: node_cluster, plot_kind: categorical_distribution}

synthesis_fields:
  - name: method
    type: str
    allowed_values_ref: allowed_synthesis_methods
    graph_role: node_cluster
```

It is used by passing its path to the scanner:

```python
scanner = ComProScanner(
    main_property_keyword="piezoelectric",
    schema_config_path="schemas/piezo_schema.yaml",
)
```

Types range from scalars (`str`, `float`, unions such as `float|None`) and lists to `dict[str, ...]` for values keyed by an arbitrary string, `object` for a fixed-shape block, and `dict[str, object]` for blocks keyed by an arbitrary string. Because `object` and `dict[str, object]` fields contain further fields, a schema can describe a tree of any depth: in the full example on the [Custom Schema](../../custom-schema.md) page, each composition maps to a measurement, its conditions, a microstructure with grain-size measurements keyed by imaging technique, and a crystal phase.

Three rules keep the format predictable:

- **A custom schema replaces the default**; nothing is merged. The one exception is `abbreviations`, which is attached automatically if missing, because the formatting agent needs it to expand abbreviated compositions into full formulas.
- **The output envelope stays fixed.** Records keep `composition_data`, `synthesis_data` and `article_metadata`; new fields go inside the first two. Supporting further top-level sections would need separate extraction crews, a different score split and a defined place in the graph, so it was deliberately left out.
- **Schemas are validated on load.** Typos in attribute names, unsupported types, weights that exceed 1, prompt placeholders a crew cannot fill and unsafe graph identifiers are all reported before a single article is processed.

---

## 3. Extraction: Generated Prompts and Models

The four hand-written output models were replaced by models generated from the schema, with nested fields becoming nested models. Before deleting the old models, their JSON schemas, default values, field annotations and the model descriptions CrewAI generates from them were captured; the generated models match them on all five.

The prompts no longer name any field. Each crew renders three placeholders from the schema: a field description (types, required or optional, descriptions and, for the formatting agents, per-field formatting rules and allowed values), an example JSON built from the fields' `example` values, and the list of field names. Field-specific guidance, such as "return an empty list if the precursors are not clearly mentioned", moved from the prompt files into the fields' descriptions, so it disappears with the field. The default few-shot examples are now generated as well, which fixed the `"compositions"` drift mentioned above.

---

## 4. Evaluation: Leaf by Leaf

The most important decision for evaluation was to keep the scoring formula unchanged. Each section still starts from 1.0 and subtracts weighted penalties, the composition and synthesis sections still count for half of the score each, and true/false positives and negatives are counted as before. What the schema adds is *where the weights and comparison rules come from*.

**Match modes.** Every field compares its values with one of six existing rules: `exact`, `semantic` (embedding similarity), `numeric` (with a tolerance), `item_list` (lists matched item by item), `joined_text` (a list compared as one paragraph) or `keyed_values` (composition-keyed values). When a field doesn't set one, it is derived from the type.

**Cascading weights.** Weights are given relative to sibling fields and multiplied down the tree, so each leaf receives an effective weight, and the leaves of a section always sum to 1. Fields without a weight share what their siblings leave. The default schema's weights reproduce the previous evaluator weights bit for bit.

**Leaf-level items.** Previously, a composition's value was compared as a whole, which was fine for a single number. For nested values this would make one wrong digit in a grain size fail the entire record of a composition. Now every leaf is one item with its own tolerance, and each entry of a nested dictionary is a separate item: the SEM and EBSD grain sizes of a composition are judged independently, and a missing composition counts its key plus every leaf as missing.

**Deterministic matching and swap correction.** While building regression baselines, the semantic evaluator turned out to be nondeterministic: composition keys that didn't match exactly were paired greedily while iterating Python sets, whose order depends on the process's hash seed, so the same paper could score 35, 37 or 39 true positives on different runs. Keys and papers are now processed in sorted order. A related problem affected near-identical compositions, such as two BCZT variants that differ only in the amount of an additive: all cross-pairings clear the similarity threshold, so greedy matching could pair c1 with c2's value and c2 with c1's. A final re-assignment step (`scipy.optimize.linear_sum_assignment`) now maximises, in order of priority, the number of matched keys, then the number of matching values, then the similarity. Putting matched keys first guarantees that a correction can only add value matches, never lose a key match. Across 1,692 stored paper evaluations, 48 changed: 5 scores went up, none went down, and 86 key pairs were corrected.

**Agentic evaluation.** The evaluation agents had their own hand-written response models, which required `family` and `property_unit`; a schema without them would have failed validation. Both the response models and the prompts are now generated from the schema, with each field's judging rule in its `evaluation_note`. For nested values the agent returns one decision per leaf. Interchanged compositions needed no special handling here, as the agent already pairs compositions itself.

---

## 5. Plots and the Knowledge Graph

Any field or nested leaf can now be plotted by its dotted path, for example `"processing.sintering.profile.segments.temperature_c"`, with `plot_categorical_distribution()` or the new `plot_numeric_distribution()`. Declaring `plot_kind` in the schema is enough for `plot_all_distributions()` to include a field. Evaluation charts accept a `"<field>_match"` metric for every top-level field.

For the knowledge graph, two representations of nested data were considered: flattening leaves into properties of the composition node (`grain_mean_size`), or following the JSON structure with separate nodes. Both can be queried with Cypher, but only the second scales to an arbitrary tree: a dictionary inside a block inside a block is the same rule applied again, while flattening needs a new naming convention at every level and has no clean representation for nested dictionaries. The graph therefore follows the JSON structure:

- a block becomes a child node owned by its parent (`HAS_<FIELD>`), never shared, so two compositions with identical d33 data keep separate nodes;
- a dictionary becomes one node per entry, identified by its key;
- a leaf becomes a property of its parent node, a shared node per distinct value (with optional semantic clustering), a joined text node, or is skipped, as its `graph_role` says.

```cypher
// compositions with SEM grain data, with their d33
MATCH (p:Paper)-[:MENTIONS]->(c:Composition)-[:HAS_MICROSTRUCTURE]->()-[:HAS_GRAIN]->()
      -[:HAS_MEASUREMENTS]->(g:GrainMeasurement {technique: "SEM"})
MATCH (c)-[:HAS_D33_DATA]->(d:D33Data)
RETURN p.doi, c.composition, d.value, g.mean
```

Labels and relationship types cannot be passed to Neo4j as query parameters, so the schema loader only accepts safe identifiers for them.

---

## 6. Verification: Nothing Changed for Existing Users

Since every published ComProScanner result was produced with the earlier code, the default schema had to reproduce it exactly. Each stage was compared against the previous code:

| Stage | Comparison | Result |
| --- | --- | --- |
| Output models | 4 generated vs hand-written models, 5 representations each | identical |
| Prompts | every crew rendered through CrewAI with real inputs, default and nested schemas | no leftover placeholders |
| Semantic evaluation | 19 stored runs with the text matcher, 3 with PhysBERT | identical within 1e-12 |
| Agentic evaluation | 13 runs replayed offline from recorded agent decisions | identical within 1e-12 |
| Distribution plots | category counts and clusters on 4 result files | identical |
| Evaluation charts | metric values and heatmap matrices on 7 evaluation results | identical |
| Knowledge graph | 3 graphs built in Neo4j by both builders (up to 2,391 nodes and 4,186 relationships) | 0 differences |

The remaining differences, below 1e-12, come from summing the same penalties in a different order. The only intended changes in results are those from the deterministic matching and swap correction described in Section 4. The full test suite passes (734 tests), including new tests for custom and nested schemas at every stage.

Validating the documentation's complex example also exposed two bugs, both fixed before release: a keyed dictionary nested *inside* a block was compared as one item by the semantic evaluator and was not walked by the plots.

---

## 7. Getting Started

1. Copy `src/comproscanner/utils/configs/default_extraction_schema.yaml`, which documents every attribute in its header.
2. Add, remove or nest fields, giving each a description and an example.
3. Pass the file as `schema_config_path` to `ComProScanner`, and to the plotting and knowledge-graph functions.

The [Custom Schema](../../custom-schema.md) page has the full attribute reference and a complete five-level example with an extracted record, its evaluation, plots and graph.

---

## References

1. A. Roy, E. Grisan, J. Buckeridge and C. Gattinoni, Digital Discovery, 2026, 5, 1794-1808.
