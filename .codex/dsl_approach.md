 # Domain-Specific Language (DSL) for Custom Workflows

 This document outlines a proposed **DSL approach** to let end users compose custom analysis workflows by combining our built-in `Process` modules in an expressive, declarative style.

 ## Goals

 - Provide a **fluent, chainable API** for users to declare pipelines in code (or via YAML/JSON).
 - Leverage existing `Process` classes (`ThemeGeneration`, `ThemeAllocation`, `ThemeExtraction`, `SentimentProcess`, `Cluster`, etc.).
 - Maintain **dependency resolution** automatically (topological sorting of steps based on `depends_on`).
 - Reuse the high-level `Analyzer` engine under the hood (caching, execution, result wrapping).
 - Offer multiple DSL surfaces:
   - Python fluent API (method chaining / operator overloading)
   - Configuration-based (YAML / JSON) pipelines
   - CLI integration for workflow files

 ## Python Fluent API

 ```python
 from pulse.dsl import Workflow

 # Build a pipeline by chaining steps (default inputs come from original dataset or prior outputs):
 wf = (
     Workflow()
     .theme_generation(min_themes=2, max_themes=5, fast=True)
     .theme_allocation(threshold=0.6)
     .theme_extraction()
     # Compute sentiment on the original texts
     .sentiment(fast=True, source='dataset')
     # Compute sentiment on extracted elements
     .sentiment(fast=True, source='theme_extraction')
     # Cluster the generated themes
     .cluster(k=3, source='theme_generation')
 )

 # Execute on a list of texts
 results = wf.run(texts)

 # Access individual results
 df_alloc = results.theme_allocation.to_dataframe()
 df_extr  = results.theme_extraction.to_dataframe()
 summary  = results.sentiment.summary()
 labels   = results.cluster.kmeans(n_clusters=3)
 ```

 ### Under the hood

 - Each DSL method appends a `Process` instance to an internal list.
 - Dependencies are inferred via `Process.depends_on` or explicitly overridden.
 - `Workflow.run(...)` constructs an `Analyzer` with the collected `processes` and calls `.run()`.

 ## Configuration File (YAML / JSON)

 Users can define pipelines in a file:

 ```yaml
 pipeline:
   - theme_generation:
       min_themes: 3
       max_themes: 7
       fast: true
   - theme_allocation:
       threshold: 0.5
   - theme_extraction: {}
   # Sentiment on original dataset
   - sentiment:
       source: dataset       # 'dataset', 'theme_generation', 'theme_extraction', etc.
       fast: false
   # Cluster the generated themes
   - cluster:
       k: 4
       source: theme_generation
 ```

 Then execute via Python or CLI:

 ```python
 from pulse.dsl import Workflow

 wf = Workflow.from_file("my_pipeline.yaml")
 results = wf.run(texts)
 ```

 ## CLI Integration

 - `pulse workflow run --file my_pipeline.yaml --input data.txt`
 - Progress and summaries printed to console.

 ## Next Steps

 1. Implement a `Workflow` builder class with:
    - Method chaining
    - `__rshift__` or `|` operator overloading for pipeline definition
    - `from_file` constructor for YAML/JSON
 2. Extend CLI entry point (`pulse workflow`) to parse config files.
 3. Add documentation and examples in `docs/` and the README.

 This DSL layer reuses our core implementation (models, client, analyzer), minimizing duplication while offering a more declarative user experience.

## Managing Multiple Named Data Sources and Inputs

In this advanced model, users can register and wire multiple named data streams to processes, offering full control over inputs and dependencies:

1. **Named DataStreams**
   - Users declare sources explicitly:
     ```python
     wf = (
         Workflow()
         .source("comments", comments_list)
         .source("themes", existing_themes_list)
         ...
     )
     ```
   - Processes then reference these by name or by downstream outputs.

2. **Type Metadata and Validation**
   - Under the hood, each stream carries lightweight metadata (e.g. `List[str]`, `List[Theme]`).
   - Processes declare accepted input types and emitted types.
   - The DSL engine performs upfront validation of types and shapes, failing fast on mismatches.

3. **Flexible `source` API Per Process**
   - Processes can consume one or multiple named streams:
     ```python
     .theme_allocation(inputs="comments", themes_from="themes")
     .theme_allocation(inputs="comments", themes_from="theme_generation")
     ```
   - If no explicit `inputs` are provided, defaults are inferred (e.g. original dataset).

4. **DAG-Based Execution vs. Linear Chains**
   - Internally, pipelines form a directed acyclic graph (DAG) of nodes (sources + processes).
   - Edges represent dataflow, allowing parallel execution of independent branches and fine-grained caching.

5. **Python-Idiomatic and Familiar**
   - This pattern aligns with other Python workflow libraries (e.g. Prefect, Luigi).
   - Lightweight and explicit wiring avoids magic while remaining expressive.

6. **Trade-offs**
   - More upfront configuration (naming sources, typing) in exchange for flexibility and safety.
   - Once the DAG executor is in place, adding or mixing new processes is straightforward.

**Note**: This DSL extension is an **experimental parallel enhancement**. The existing high-level `Analyzer` and `Process` API will remain fully supported and unchanged. Users can adopt the DSL approach incrementally without impacting existing code.
