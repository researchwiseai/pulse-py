# DSL Builder

The DSL provides a fluent way to compose named sources and processes into a workflow with optional monitoring, custom wiring, and execution graph inspection.

Module: `pulse.dsl`

## Workflow Basics

```python
from pulse.dsl import Workflow

texts = ["I love pizza", "I hate rain"]

wf = (
    Workflow()
    .source("docs", texts)
    .theme_generation(source="docs", min_themes=2)
    .sentiment(source="docs")
)

result = wf.run(fast=True)  # or pass a CoreClient via client=...
print(result.theme_generation.themes)
print(result.sentiment.sentiments)
```

Key ideas:
- Register data sources with `.source(name, data)`.
- Add steps; each step can optionally name its input source or derive it from prior steps.
- `run()` executes steps in order and returns a container with attributes for each step id (or alias).

## Steps and Parameters

All steps accept `name=` to alias the step; subsequent steps can reference the alias.

### `.theme_generation(...)`
Parameters:
- `min_themes=2`, `max_themes=10`, `context=None`, `version=None`, `prune=None`, `fast=None`, `source: str | None = None`, `name=None`.
If `source` is omitted, uses the dataset source (see run behavior below).

### `.sentiment(...)`
Parameters:
- `fast=None`, `source=None`, `name=None`.

### `.theme_allocation(...)`
Parameters:
- `themes: list[str] | None = None`, `single_label=True`, `threshold=0.5`, `fast=None`, `inputs: str | None = None`, `themes_from: str | None = None`, `name=None`.
Notes:
- If `themes=None`, it pulls themes from the most recent `.theme_generation(...)` step, or from `themes_from` if specified.
- `inputs` selects the text source (defaults to dataset).

### `.theme_extraction(...)`
Parameters:
- `themes=None`, `version=None`, `fast=None`, `inputs=None`, `themes_from=None`, `name=None`.

### `.cluster(...)`
Parameters:
- `k=2`, `source=None`, `fast=None`, `name=None`.

## Running a Workflow

```python
result = wf.run(dataset=texts, fast=True)
```

- If you registered any `.source(...)`, `run()` operates in DSL mode:
  - Optional `dataset` is registered under the implicit source `"dataset"` if not already set.
  - Each process reads inputs from its wired `_inputs` (default `"dataset"`).
  - A `CoreClient` is created if one is not provided.
- If you did not register any `.source(...)`, `run(dataset=..., ...)` delegates to the `Analyzer` with the accumulated processes.

## Monitoring

Register lifecycle hooks for observability:

```python
def on_run_start(): ...
def on_process_start(pid): ...
def on_process_end(pid, result): ...
def on_run_end(): ...

wf = Workflow().monitor(
  on_run_start=on_run_start,
  on_process_start=on_process_start,
  on_process_end=on_process_end,
  on_run_end=on_run_end,
)
```

## From File

You can build workflows from JSON or YAML files with a top‑level `pipeline` array, where each entry is a single‑key map of step name to parameters:

```yaml
pipeline:
  - theme_generation: { min_themes: 2, max_themes: 5 }
  - sentiment: {}
```

Load and run:
```python
wf = Workflow.from_file("pipeline.yml")
res = wf.run(dataset=texts)
```

## Nested Inputs (Sentiment)

For `.sentiment(...)`, DSL supports nested list inputs by flattening before the API call and reconstructing the nested shape on output. Other steps operate on flat lists.

## Graph

`graph()` returns an adjacency list describing the execution DAG, including static dependencies and wired inputs. This helps visualize or validate your pipeline.

```python
edges = wf.graph()
for node, deps in edges.items():
    print(node, "<-", deps)
```

