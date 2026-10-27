# Custom Schema

A custom schema lets you decide **what ComProScanner extracts**, field by field, in a single YAML file and without editing any Python code. The same file then drives every later stage: the extraction prompts and output validation, data cleaning, both evaluators, the distribution plots and the Neo4j knowledge graph.

!!! note "You don't need a schema to get started"

    When no schema is given, ComProScanner uses its bundled default schema, which reproduces the classic output: `compositions_property_values`, `property_unit`, `family`, `method`, `precursors`, `steps` and `characterization_techniques`. A custom schema is only needed when you want different or additional fields.

## Using a Custom Schema

Pass the path to your YAML file when creating the scanner. Extraction, data cleaning and both evaluations of that scanner then use it:

```python
from comproscanner import ComProScanner

scanner = ComProScanner(
    main_property_keyword="piezoelectric",
    schema_config_path="schemas/piezo_schema.yaml",
)

scanner.extract_composition_property_data(main_extraction_keyword="d33")
scanner.clean_data(json_results_file="results.json")
scanner.evaluate_semantic(
    ground_truth_file="ground_truth.json",
    test_data_file="results.json",
    extraction_agent_model_name="gpt-4o-mini",
)
```

The plotting and knowledge-graph functions are module-level, so they take the schema path as an argument:

```python
from comproscanner import data_visualizer, eval_visualizer

data_visualizer.plot_all_distributions(
    data_sources="results.json",
    output_folder="plots",
    schema_config_path="schemas/piezo_schema.yaml",
)
data_visualizer.create_knowledge_graph(
    result_file="results.json",
    schema_config_path="schemas/piezo_schema.yaml",
)
eval_visualizer.plot_single_bar_chart(
    result_file="semantic_evaluation_result.json",
    metrics_to_include=["overall_accuracy", "phase_match"],
    schema_config_path="schemas/piezo_schema.yaml",
)
```

!!! warning "A custom schema replaces the default"

    Your file replaces the default schema entirely; nothing is merged. A field from the default schema exists in your output only if your file declares it. The one exception is `abbreviations`: if your schema doesn't declare it, the default `abbreviations` field is added automatically, because the formatting agent needs it to expand abbreviated compositions into full formulas.

!!! info "The output envelope stays fixed"

    Every extracted record keeps three top-level keys: `composition_data`, `synthesis_data` and `article_metadata`. Your fields go inside `composition_data` and `synthesis_data`, as many as you like and nested as deep as you like. The schema format also accepts an `extra_domains` key, but it currently has no effect.

The best starting point is a copy of the bundled default, `src/comproscanner/utils/configs/default_extraction_schema.yaml`, which also documents every field attribute in its header.

## Schema File Structure

| Key | Required | Description |
| --- | --- | --- |
| `version` | yes | Schema format version. Currently `"1.0"`. |
| `composition_fields` | yes | Fields of the `composition_data` section. |
| `synthesis_fields` | yes | Fields of the `synthesis_data` section. |
| `primary_composition_field` | no | The composition-keyed field used for data cleaning, for deciding whether a paper has usable data, and as the `Composition` node of the knowledge graph. Defaults to the first `dict`-typed composition field. |
| `extra_domains` | no | Parsed and validated, but not used by any stage. |

### Field Types

| Type | Meaning | Example value |
| --- | --- | --- |
| `str`, `int`, `float`, `bool` | A single value | `"pC/N"`, `620` |
| `float\|None` | A union: either type (combine any types with `\|`) | `97.2` or `null` |
| `list[str]`, `list[float]` | A list | `["BaCO3", "TiO2"]` |
| `dict[str, float\|None]` | Values keyed by an arbitrary string | `{"BaTiO3": 190}` |
| `object` | A fixed-shape block whose fields are listed in `value_fields` | `{"value": 620, "unit": "pC/N"}` |
| `dict[str, object]` | Blocks keyed by an arbitrary string, each shaped by `value_fields` | `{"SEM": {"mean": 18.5}}` |

`object` and `dict[str, object]` fields can contain further `object` and `dict[str, object]` fields, so a schema can be nested to any depth.

## Field Attributes

Every field has a `name` (a lowercase identifier such as `sintering_temp`) and a `type`. Everything else is optional.

### Extraction Attributes

| Attribute | Default | Description |
| --- | --- | --- |
| `description` | `""` | Guidance shown to the extraction agents. Composition fields may use `{main_extraction_keyword}`, which is replaced by your property keyword; synthesis fields can't use any placeholder. |
| `required` | `true` | Shown to the agents as required or optional. Missing values never fail validation; they take the default. |
| `default` | from the type | Value used when the field is missing (`""` for text, `[]` for lists, `{}` for dictionaries, `null` for numbers and blocks). |
| `example` | type placeholder | Example value used in the example JSON shown to the agents. |
| `variable_example` | `example` | Example value for the variable-composition example (composition keys written with variables such as `x`). |
| `format_note` | none | A rule for the formatting agent only, e.g. "Keep the steps unmodified." |
| `allowed_values` | none | A list of allowed values the formatting agent maps this field onto. |
| `allowed_values_ref` | none | Synthesis fields only: `allowed_synthesis_methods` or `allowed_characterization_techniques`, to use the lists passed to `extract_composition_property_data()`. |
| `extraction_only` | `false` | Kept by the extraction agent but dropped by the formatting agent (this is how `abbreviations` works). |

### Evaluation Attributes

| Attribute | Default | Description |
| --- | --- | --- |
| `weight` | equal share | Importance relative to sibling fields. Explicit weights in a group may sum to at most 1; fields without a weight share the remainder. |
| `match_mode` | from the type | How values are compared: `exact`, `semantic`, `item_list`, `joined_text`, `keyed_values` or `numeric` (see below). |
| `similarity_threshold` | per mode | Matching threshold: 0.9 for `semantic`, `item_list` and `keyed_values` keys, 0.8 for `joined_text`. For `numeric` fields it is a tolerance; without it, `value_error_thresholds` (or exact equality) applies. |
| `similarity_threshold_mode` | `relative` | For numeric tolerances: `relative` (a fraction of the reference value) or `absolute` (in the field's own units). |
| `evaluation_note` | generic rule | Judging rule given to the agentic evaluator for this field. |

| `match_mode` | Comparison | Derived for |
| --- | --- | --- |
| `exact` | Values must be identical | `bool` |
| `semantic` | Embedding similarity at least the threshold | `str` |
| `numeric` | Within the tolerance; lists are compared element by element | `int`, `float`, `list[float]` |
| `item_list` | Lists matched item by item (similarity tiers 0.9 and 0.7) | `list[str]` |
| `joined_text` | List joined into one paragraph and compared as text | (set explicitly, e.g. for `steps`) |
| `keyed_values` | Keys matched exactly, then by similarity with swap correction; values compared leaf by leaf | `dict[...]` |

### Plotting Attributes

| Attribute | Default | Description |
| --- | --- | --- |
| `plot_kind` | `none` | `categorical_distribution` (bar or pie chart of values), `numeric_distribution` (histogram) or `none`. Used by `plot_all_distributions()`. |

### Knowledge Graph Attributes

| Attribute | Default | Description |
| --- | --- | --- |
| `graph_role` | `node_property` | For leaf fields: `node_property` (a property of the parent node), `node_cluster` (a shared node per distinct value), `joined_node` (a list joined into one text node) or `skip`. Blocks and dictionaries are always nodes. |
| `graph_label` | field name in CamelCase | Neo4j label of the field's nodes. |
| `relationship_type` | `HAS_<FIELD>`, or `USES_<FIELD>` for clusters | Relationship from the parent node. |
| `graph_edge_sources` | none | For a composition-level cluster: relationship type per source node, e.g. `{paper: BELONGS_TO_FAMILY, composition: BELONGS_TO}`. |
| `graph_key_property` | `key` | Property holding a dictionary entry's key. |
| `graph_value_property` | `value` | Property holding a dictionary entry's single value, or a joined node's text. |
| `graph_default` | none | Value used when the field is missing (e.g. `UNKNOWN_FAMILY`). |
| `cluster_similarity_threshold` | 0.85 | Threshold for semantically merging similar values of a `node_cluster` field. |

!!! tip "Quote text containing commas in inline YAML"

    In the compact `{name: ..., type: ...}` style, an unquoted comma ends the value, so `description: Grain size unit, e.g. um` breaks the field. Quote such text: `description: "Grain size unit, e.g. um"`. The schema loader reports an error if this happens.

## Complete Example: A Multi-Level Schema

This schema extracts rich piezoceramic data. Each composition's value is a tree up to five levels deep (d33 measurement → measurement details → conditions; microstructure → grain → measurements per imaging technique), and both sections gain new keys: `dopants` under `composition_data`, and a nested `processing` block under `synthesis_data`.

??? example "piezo_schema.yaml"

    ```yaml
    version: "1.0"
    primary_composition_field: compositions_property_values

    composition_fields:
      - name: compositions_property_values
        type: "dict[str, object]"
        description: >-
          Every material composition reported in the article's own work, mapped to its measured data.
          Keys are full chemical formulas with proper fractions.
        weight: 0.7
        graph_label: Composition
        relationship_type: MENTIONS
        graph_key_property: composition
        value_fields:
          - name: d33_data
            type: object
            description: The {main_extraction_keyword} measurement for this composition.
            weight: 0.5
            value_fields:
              - name: value
                type: "float|int|None"
                description: The numeric {main_extraction_keyword} value.
                weight: 0.7
                similarity_threshold: 0.05
                plot_kind: numeric_distribution
                example: 620
              - name: unit
                type: str
                description: The unit of the value, e.g. pC/N or pm/V.
                match_mode: exact
                plot_kind: categorical_distribution
                example: pC/N
              - name: measurement
                type: object
                description: How the value was measured.
                required: false
                value_fields:
                  - name: method
                    type: str
                    description: The measurement technique, e.g. Berlincourt d33 meter or laser interferometry.
                    graph_role: node_cluster
                    graph_label: MeasurementMethod
                    relationship_type: MEASURED_BY
                    example: Berlincourt d33 meter
                  - name: conditions
                    type: object
                    description: Conditions under which the value was measured.
                    required: false
                    value_fields:
                      - {name: temperature_c, type: "float|None", description: Measurement temperature in degrees Celsius., example: 25}
                      - {name: poling_field_kv_per_mm, type: "float|None", description: Poling electric field in kV/mm., example: 3.0}
          - name: microstructure
            type: object
            description: Microstructure of this composition.
            required: false
            weight: 0.3
            value_fields:
              - name: relative_density_percent
                type: "float|None"
                description: Relative density as a percentage of the theoretical density.
                similarity_threshold: 1.0
                similarity_threshold_mode: absolute
                plot_kind: numeric_distribution
                example: 97.2
              - name: grain
                type: object
                description: Grain size data.
                required: false
                value_fields:
                  - {name: unit, type: str, description: "Grain size unit, e.g. um or nm.", match_mode: exact, example: um}
                  - name: measurements
                    type: "dict[str, object]"
                    description: Grain size measurements keyed by the imaging technique (e.g. SEM, EBSD).
                    graph_label: GrainMeasurement
                    graph_key_property: technique
                    value_fields:
                      - name: mean
                        type: "float|None"
                        description: Mean grain size.
                        similarity_threshold: 0.1
                        plot_kind: numeric_distribution
                        example: 18.5
                      - name: sizes
                        type: "list[float]"
                        description: Every reported grain size.
                        similarity_threshold: 0.5
                        similarity_threshold_mode: absolute
                        example: [12.1, 17.8, 25.6]
          - name: phase
            type: object
            description: Crystal phase of this composition.
            required: false
            weight: 0.2
            value_fields:
              - name: crystal_structure
                type: str
                description: Crystal system, e.g. tetragonal or orthorhombic.
                graph_role: node_cluster
                graph_label: CrystalStructure
                relationship_type: HAS_STRUCTURE
                plot_kind: categorical_distribution
                example: tetragonal
              - {name: space_group, type: str, description: Space group symbol., match_mode: exact, example: P4mm}

      - name: property_unit
        type: str
        description: The unit in which the {main_extraction_keyword} values are reported.
        weight: 0.1
        match_mode: exact
        example: pC/N

      - name: family
        type: str
        description: The material family (base or host material system) the compositions belong to.
        weight: 0.1
        match_mode: exact
        graph_role: node_cluster
        graph_label: Family
        graph_edge_sources: {paper: BELONGS_TO_FAMILY, composition: BELONGS_TO}
        graph_default: UNKNOWN_FAMILY
        plot_kind: categorical_distribution
        example: BCZT

      - name: dopants
        type: "list[str]"
        description: Dopant elements or compounds added to the base material.
        weight: 0.1
        graph_role: node_cluster
        graph_label: Dopant
        relationship_type: DOPED_WITH
        plot_kind: categorical_distribution
        example: [Ca, Zr]

    synthesis_fields:
      - name: method
        type: str
        description: The experimental synthesis method, e.g. solid-state reaction or sol-gel.
        weight: 0.15
        allowed_values_ref: allowed_synthesis_methods
        format_note: If allowed values are listed, change the method to the matching allowed value; otherwise keep it as it is.
        graph_role: node_cluster
        graph_label: Method
        graph_default: UNKNOWN_METHOD
        plot_kind: categorical_distribution
        example: solid-state reaction

      - name: precursors
        type: "list[str]"
        description: Starting chemicals, as chemical formulas. Return an empty list if they are not clearly mentioned.
        weight: 0.15
        graph_role: node_cluster
        graph_label: Precursor
        relationship_type: USED_PRECURSOR
        plot_kind: categorical_distribution
        example: [BaCO3, CaCO3, TiO2, ZrO2]

      - name: processing
        type: object
        description: Thermal processing of the samples.
        weight: 0.4
        value_fields:
          - name: calcination
            type: object
            description: Calcination step.
            required: false
            value_fields:
              - {name: temperature_c, type: "float|None", description: Calcination temperature in degrees Celsius., plot_kind: numeric_distribution, example: 1250}
              - {name: duration_h, type: "float|None", description: Calcination time in hours., example: 4}
          - name: sintering
            type: object
            description: Sintering step.
            value_fields:
              - name: technique
                type: str
                description: Sintering technique, e.g. conventional, two-step or spark plasma sintering.
                graph_role: node_cluster
                graph_label: SinteringTechnique
                relationship_type: USES_TECHNIQUE
                plot_kind: categorical_distribution
                example: two-step sintering
              - name: atmosphere
                type: str
                description: Sintering atmosphere, e.g. air or oxygen. Leave empty if not stated.
                allowed_values: [air, oxygen, nitrogen, argon, vacuum]
                format_note: Map the atmosphere to one of the allowed values.
                graph_role: node_cluster
                graph_label: Atmosphere
                relationship_type: SINTERED_IN
                example: air
              - name: profile
                type: object
                description: Temperature profile.
                required: false
                value_fields:
                  - {name: ramp_rate_c_per_min, type: "float|None", description: Heating rate in degrees Celsius per minute., example: 5}
                  - name: segments
                    type: "dict[str, object]"
                    description: Dwell segments keyed by stage name (e.g. first_step, second_step).
                    graph_label: SinteringSegment
                    graph_key_property: stage
                    value_fields:
                      - {name: temperature_c, type: "float|None", description: Dwell temperature in degrees Celsius., plot_kind: numeric_distribution, example: 1500}
                      - {name: dwell_h, type: "float|None", description: Dwell time in hours., example: 0.1}

      - name: characterization_techniques
        type: "list[str]"
        description: Characterization techniques, as the instrument name or its short name, e.g. XRD, SEM, TEM.
        weight: 0.15
        allowed_values_ref: allowed_characterization_techniques
        graph_role: node_cluster
        graph_label: CharacterizationTechnique
        relationship_type: USED_CHARACTERIZATION_TECHNIQUE
        plot_kind: categorical_distribution
        example: [XRD, SEM, Raman Spectroscopy]

      - name: steps
        type: "list[str]"
        description: The experimental synthesis procedure as an ordered list of steps.
        weight: 0.15
        format_note: Keep the steps unmodified.
        match_mode: joined_text
        graph_role: joined_node
        graph_label: Step
        relationship_type: USED_SYNTHESIS_STEPS
        graph_value_property: steps
        graph_default: No steps available
        example: [Ball-mill the precursors in ethanol for 24 h, Calcine at 1250 °C for 4 h]
    ```

### Extracted Record

An extracted record following this schema looks like this (the `article_metadata` section is unchanged and omitted here):

??? example "Record for one paper"

    ```json
    {
      "composition_data": {
        "compositions_property_values": {
          "Ba0.85Ca0.15Ti0.9Zr0.1O3": {
            "d33_data": {
              "value": 620,
              "unit": "pC/N",
              "measurement": {
                "method": "Berlincourt d33 meter",
                "conditions": {"temperature_c": 25, "poling_field_kv_per_mm": 3.0}
              }
            },
            "microstructure": {
              "relative_density_percent": 97.2,
              "grain": {
                "unit": "um",
                "measurements": {
                  "SEM": {"mean": 18.5, "sizes": [12.1, 17.8, 25.6]},
                  "EBSD": {"mean": 16.9, "sizes": [11.4, 16.2, 23.0]}
                }
              }
            },
            "phase": {"crystal_structure": "tetragonal", "space_group": "P4mm"}
          },
          "Ba0.85Ca0.15Ti0.92Zr0.08O3": {
            "d33_data": {"value": 540, "unit": "pC/N"},
            "phase": {"crystal_structure": "orthorhombic", "space_group": "Amm2"}
          }
        },
        "property_unit": "pC/N",
        "family": "BCZT",
        "dopants": ["Ca", "Zr"]
      },
      "synthesis_data": {
        "method": "solid-state reaction",
        "precursors": ["BaCO3", "CaCO3", "TiO2", "ZrO2"],
        "processing": {
          "calcination": {"temperature_c": 1250, "duration_h": 4},
          "sintering": {
            "technique": "two-step sintering",
            "atmosphere": "air",
            "profile": {
              "ramp_rate_c_per_min": 5,
              "segments": {
                "first_step": {"temperature_c": 1500, "dwell_h": 0.1},
                "second_step": {"temperature_c": 1400, "dwell_h": 10}
              }
            }
          }
        },
        "characterization_techniques": ["XRD", "SEM", "EBSD", "d33 meter"],
        "steps": [
          "Ball-mill the precursors in ethanol for 24 h",
          "Calcine at 1250 °C for 4 h",
          "Press pellets and sinter at 1500 °C for 6 min, then at 1400 °C for 10 h"
        ]
      }
    }
    ```

Optional blocks can be left out: the second composition has no `microstructure` and no `measurement` details, and that is valid.

### Extraction

The extraction and formatting agents are shown a generated description of every field, with types, required/optional flags and your descriptions, plus an example JSON built from the `example` values. Formatting agents also see each field's `format_note` and allowed values. For instance, the synthesis formatter receives:

```text
- method (str, required): The experimental synthesis method, e.g. solid-state reaction or sol-gel.
  Formatting rule: If allowed values are listed, change the method to the matching allowed value; otherwise keep it as it is.
  Allowed values:
  <your allowed_synthesis_methods>
- processing (object, required): Thermal processing of the samples.
  An object with these fields:
    - sintering (object, required): Sintering step.
      An object with these fields:
        - atmosphere (str, required): Sintering atmosphere, e.g. air or oxygen. Leave empty if not stated.
          Formatting rule: Map the atmosphere to one of the allowed values.
          Allowed values: air, oxygen, nitrogen, argon, vacuum
```

The agents' output is validated against models generated from the same schema, so the shape of every record follows the YAML.

### Evaluation

**Weights.** Weights are normalised within each group of sibling fields and then multiplied down the tree, so each leaf receives an *effective weight*. Within a section the leaves' effective weights always sum to 1. In this schema, `d33_data.value` gets 0.7 × 0.5 × 0.7 = 0.245 of the composition score, while each grain-measurement leaf gets about 0.026. A field without an explicit weight shares whatever its siblings leave; if the explicit weights of a group already sum to 1, unweighted siblings get 0 and are not scored.

**Leaf-level items.** Nested values are compared leaf by leaf, and each leaf counts as one true/false positive/negative item. Entries of nested dictionaries are separate items too: `microstructure.grain.measurements.SEM.mean` and `….EBSD.mean` are judged independently, and an imaging technique that appears only in the extracted data counts as an extra item. If the extracted SEM mean grain size were 21.0 instead of 18.5 (outside the 10 % relative tolerance), only that one leaf would fail.

**Tolerances.** `d33_data.value` uses a 5 % relative tolerance, so an extracted 600 for a reference of 620 still matches. `relative_density_percent` uses an absolute tolerance of ±1.0 percentage points.

The agentic evaluator receives the same field list. Its prompt and response format are generated from the schema, and for nested composition values it is asked for one decision per leaf.

### Visualization

`plot_all_distributions()` draws one chart for every field or leaf with a `plot_kind`, here 13 plots, including histograms of the d33 values, mean grain sizes and sintering-segment temperatures. Any single field or leaf can also be plotted by its dotted path:

```python
from comproscanner import data_visualizer

data_visualizer.plot_numeric_distribution(
    "processing.sintering.profile.segments.temperature_c",
    data_sources="results.json",
    schema_config_path="schemas/piezo_schema.yaml",
)
data_visualizer.plot_categorical_distribution(
    "compositions_property_values.phase.crystal_structure",
    chart_type="pie",
    data_sources="results.json",
    schema_config_path="schemas/piezo_schema.yaml",
)
```

In evaluation charts, any field's match score is available as `"<field>_match"` (for example `"phase_match"` above).

### Knowledge Graph

The graph follows the JSON structure, to any depth:

- A **block** (`object`) becomes a child node owned by its parent, linked by `HAS_<FIELD>`. Blocks are never shared: two compositions with identical d33 data get two separate `D33Data` nodes.
- A **dictionary** becomes one node per entry, identified by its key (`GrainMeasurement {technique: "SEM"}`).
- A **leaf** becomes a property of its parent node, unless its `graph_role` says otherwise. `node_cluster` leaves become shared nodes, so every composition measured with a Berlincourt meter points to the same `MeasurementMethod` node.
- Composition-level fields hang off each `Composition` node; synthesis fields hang off the `Paper` node.

For this schema the graph contains, among others:

```text
(:Paper)-[:MENTIONS]->(:Composition {composition, doi, property_unit})
(:Composition)-[:HAS_D33_DATA]->(:D33Data {value, unit})-[:HAS_MEASUREMENT]->(:Measurement)-[:HAS_CONDITIONS]->(:Conditions {temperature_c, poling_field_kv_per_mm})
(:Measurement)-[:MEASURED_BY]->(:MeasurementMethod {name})
(:Composition)-[:HAS_MICROSTRUCTURE]->(:Microstructure {relative_density_percent})-[:HAS_GRAIN]->(:Grain {unit})-[:HAS_MEASUREMENTS]->(:GrainMeasurement {technique, mean, sizes})
(:Composition)-[:HAS_PHASE]->(:Phase {space_group})-[:HAS_STRUCTURE]->(:CrystalStructure {name})
(:Composition)-[:BELONGS_TO]->(:Family {name})<-[:BELONGS_TO_FAMILY]-(:Paper)
(:Composition)-[:DOPED_WITH]->(:Dopant {name})
(:Paper)-[:HAS_PROCESSING]->(:Processing)-[:HAS_SINTERING]->(:Sintering)-[:HAS_PROFILE]->(:Profile {ramp_rate_c_per_min})-[:HAS_SEGMENTS]->(:SinteringSegment {stage, temperature_c, dwell_h})
(:Sintering)-[:SINTERED_IN]->(:Atmosphere {name})
(:Paper)-[:USES_METHOD]->(:Method {name})
```

Which then supports queries such as:

```cypher
// compositions with SEM grain data, with their d33
MATCH (p:Paper)-[:MENTIONS]->(c:Composition)-[:HAS_MICROSTRUCTURE]->()-[:HAS_GRAIN]->()-[:HAS_MEASUREMENTS]->(g:GrainMeasurement {technique: "SEM"})
MATCH (c)-[:HAS_D33_DATA]->(d:D33Data)
RETURN p.doi, c.composition, d.value, g.mean

// sintering profiles of papers using two-step sintering
MATCH (p:Paper)-[:HAS_PROCESSING]->()-[:HAS_SINTERING]->(s)-[:USES_TECHNIQUE]->(:SinteringTechnique {name: "two-step sintering"})
MATCH (s)-[:HAS_PROFILE]->()-[:HAS_SEGMENTS]->(seg)
RETURN p.doi, seg.stage, seg.temperature_c, seg.dwell_h
```

!!! tip "Seeing the whole graph in Neo4j Browser"

    `MATCH (n) RETURN n` returns only nodes, so relationships are drawn only when "Connect result nodes" is enabled in the viewer's settings. To see everything regardless of the setting, return the relationships too: `MATCH (n)-[r]->(m) RETURN n, r, m`.

## Validation

The schema file is checked when it is loaded, so mistakes surface before any article is processed. The loader reports, among others:

- unknown keys in a field (typos such as `descripton`) and unsupported types
- field names that aren't lowercase identifiers, and duplicate names
- `object` fields without `value_fields`
- explicit weights in one group that sum to more than 1
- placeholders a crew can't fill (anything other than `{main_extraction_keyword}` in composition descriptions)
- graph labels, relationship types and property names that aren't safe identifiers, since they are written into Cypher queries

## Next Steps

- Configure [RAG](rag-config.md)
- Run [Data Extraction](usage/data-extraction.md)
- Learn about [Evaluation](usage/evaluation/overview.md)
- Explore [Visualization](usage/visualization/overview.md)
