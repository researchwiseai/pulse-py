# pulse-sdk
[![Deploy Docs to GitHub Pages](https://github.com/researchwiseai/pulse-py/actions/workflows/docs.yml/badge.svg)](https://github.com/researchwiseai/pulse-py/actions/workflows/docs.yml)

Idiomatic, type-safe Python client for the Researchwise AI Pulse REST API.

## Features
- Low‑level CoreClient for direct API calls: embeddings, similarity, themes, clustering, sentiment, summaries, extractions
- Usage reporting surfaced on all responses (`resp.usage_total`, `resp.usage_records_by_feature()`)
- High‑level Analyzer for orchestrating multi‑step workflows with caching
- Built-in processes: ThemeGeneration, ThemeAllocation, SentimentProcess, Cluster
- Result helpers: pandas DataFrame conversion, summaries, visualizations (bar charts, scatter, dendrogram)
- On‑disk and in‑memory caching via diskcache
- First-class interop with pandas, NumPy, and scikit‑learn

## Documentation

- Online docs: https://researchwiseai.github.io/pulse-py/
- In-repo docs: see `docs/README.md` for the index.
- Build with MkDocs:
  - Install: `pip install mkdocs mkdocs-material`
  - Serve locally: `mkdocs serve` (http://127.0.0.1:8000)
  - Build static site: `mkdocs build`
 
## First-Time Setup (Developers)

Use Python 3.8+ and a virtual environment.

1) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
```

2) Install dependencies (SDK + dev tools)
```bash
pip install -e ".[dev]"
```

3) Install pre-commit hooks
```bash
pre-commit install
# optional: run once on all files
pre-commit run --all-files
```

4) Run tests
```bash
make test
# or
pytest
```

5) Re-record HTTP cassettes when needed
```bash
make vcr-record
```

6) Formatting and linting
```bash
black .
nbqa black .
ruff check pulse tests
```
 
## Installation

### From PyPI
Install the latest stable release:
```bash
pip install pulse-sdk
```

### From Source
Get the repository and install editable with developer dependencies:
```bash
git clone https://github.com/researchwiseai/pulse-py.git
cd pulse-py
python -m venv venv         # create a virtual environment (optional but recommended)
source venv/bin/activate    # on Windows use `venv\\Scripts\\activate`
pip install -e ".[dev]"        # install pulse-sdk plus dev tools (pytest, black, ruff, etc.)
pre-commit install           # set up formatting/linting on commit
```

## Getting Started

Once installed, you can quickly try out the core and DSL APIs.

### CoreClient
```python
from pulse.core.client import CoreClient

# Unauthenticated (dev) environment
client = CoreClient()  # default to dev environment
emb = client.create_embeddings(["Hello world", "Goodbye"], fast=True)
print(emb.embeddings)
print("total usage:", emb.usage_total)

# Submit a long-running job asynchronously
job = client.create_embeddings(["foo"] * 300, fast=False, await_job_result=False)
result = job.wait()
```

### CoreClient With Authentication

Secure your requests by providing an OAuth2 auth object to CoreClient:

```python
from pulse.core.client import CoreClient
from pulse.auth import ClientCredentialsAuth, AuthorizationCodePKCEAuth

# Client Credentials flow
auth = ClientCredentialsAuth(
    token_url="https://dev.core.researchwiseai.com/oauth2/token",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    scope="YOUR_SCOPE",  # optional
)
client = CoreClient(auth=auth)
resp = client.create_embeddings(["Hello world", "Goodbye"])  # will include Authorization header

# Authorization Code flow with PKCE
auth = AuthorizationCodePKCEAuth(
    token_url="https://dev.core.researchwiseai.com/oauth2/token",
    client_id="YOUR_CLIENT_ID",
    code="AUTHORIZATION_CODE",
    redirect_uri="https://yourapp/callback",
    code_verifier="YOUR_CODE_VERIFIER",
    scope="YOUR_SCOPE",  # optional
)
client = CoreClient(auth=auth)
resp = client.create_embeddings(["Hello world", "Goodbye"])
```

### Usage Reporting

All feature responses include usage information when available:

```python
resp = client.create_embeddings(["Hello world"], fast=True)
print(resp.usage_total)
for record in resp.usage.records:
    print(record.feature, record.units)
```

### Summarize Text

```python
from pulse.starters import summarize

# Works with a list of strings or a file path
summary = summarize("reviews.txt", question="What do people think?")
print(summary.summary)
```

### Generate Summary

```python
from pulse.core.client import CoreClient

client = CoreClient()
resp = client.generate_summary(
    ["Great food, slow service"],
    "What do diners mention?",
    length="short",  # optional
    preset="five-point",  # optional
    fast=True,
)
print(resp.summary)
```

### Cluster Texts

```python
from pulse.starters import cluster_analysis

# Cluster comments from a CSV file into two groups
clusters = cluster_analysis("reviews.csv", k=2)
print(clusters.clusters)
```

### Cluster Texts With CoreClient

```python
from pulse.core.client import CoreClient

client = CoreClient()
resp = client.cluster_texts(
    ["Good", "Bad", "Okay"],
    k=2,
    algorithm="skmeans",  # optional
    fast=True,
)
print(resp.clusters)
```

### Extract Elements

```python
client = CoreClient()
resp = client.extract_elements(
    texts=["The food was great and the service was slow."],
    categories=["food", "service"],
    dictionary={"food": ["food"], "service": ["service"]},  # optional
    use_ner=True,  # optional
    use_llm=False,  # optional
    fast=True,
)
print(resp.columns)
print(resp.matrix)
```

### Polling Asynchronous Jobs

```python
import time
client = CoreClient()
job = client.analyze_sentiment(["hello"], fast=False, await_job_result=False)
while True:
    status = client.get_job_status(job.id)
    if status.status == "completed":
        result = client.client.get(status.result_url).json()
        break
    time.sleep(1)
print(result)
```

`Job.result()` is an alias for `wait()` if you prefer a blocking call.

### Analyzer
```python
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess

texts = ["I love pizza", "I hate rain"]
processes = [ThemeGeneration(min_themes=2), SentimentProcess()]
with Analyzer(dataset=texts, processes=processes, cache_dir=".pulse_cache") as az:
    results = az.run()

print(results.theme_generation.to_dataframe())
print(results.sentiment.summary())
```

### DSL Builder With Monitoring

```python
from pulse.dsl import Workflow

# Example dataset
texts = ["I love pizza", "I hate rain"]

# Define lifecycle callbacks
def on_run_start():
    print("Workflow starting")

def on_process_start(process_id):
    print(f"Starting process: {process_id}")

def on_process_end(process_id, result):
    print(f"Finished process: {process_id}, result: {result}")

def on_run_end():
    print("Workflow finished")

# Build and run workflow
wf = (
    Workflow()
    .source("docs", texts)
    .theme_generation(source="docs", min_themes=2)
    .sentiment(source="docs")
    .monitor(
        on_run_start=on_run_start,
        on_process_start=on_process_start,
        on_process_end=on_process_end,
        on_run_end=on_run_end,
    )
)
results = wf.run()

# Access results
print(results.theme_generation.themes)
print(results.sentiment.sentiments)
```

### Optional Parameters

- **context** – provide additional context or focus for `generate_themes`.
- **version** – lock API calls (e.g., `analyze_sentiment`, `generate_themes`) to a specific model version.
- **algorithm** – choose the clustering algorithm in `cluster_texts`/`cluster_analysis`.
- **length** and **preset** – control output style in `generate_summary`.

## Examples
You can find Jupyter notebooks demonstrating both the high-level and DSL APIs under the `examples/` directory:
```bash
jupyter notebook examples/high_level_api.ipynb
jupyter notebook examples/dsl_api.ipynb
```

## Environment Variables
For authenticated access and test recording/playback, configure the following environment variables:

- `PULSE_CLIENT_ID`: your OAuth2 client ID (e.g., Auth0 client ID).
- `PULSE_CLIENT_SECRET`: your OAuth2 client secret.
- `PULSE_TOKEN_URL` (optional): token endpoint URL. Defaults to `https://wise-dev.eu.auth0.com/oauth/token`.
- `PULSE_AUDIENCE` (optional): API audience URL. Defaults to `https://dev.core.researchwiseai.com/pulse/v1`.

In local development, you can export these variables:
```bash
export PULSE_CLIENT_ID="your_client_id"
export PULSE_CLIENT_SECRET="your_client_secret"
```

In CI (e.g., GitHub Actions), add these values as repository secrets and reference them in your workflow:
```yaml
env:
  PULSE_CLIENT_ID: ${{ secrets.PULSE_CLIENT_ID }}
  PULSE_CLIENT_SECRET: ${{ secrets.PULSE_CLIENT_SECRET }}
```

## Development & Contributing

### Local Dev Setup
Note: For onboarding, see First-Time Setup above.
- Use Python 3.8+.
- Create and activate a virtual environment, then install dev deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -e .[dev]
  ```
- Install pre-commit hooks (auto-runs formatters/linters on commit):
  ```bash
  pre-commit install
  # optional: run hooks on all files once
  pre-commit run --all-files
  ```

### Format & Lint
- Format Python: `black .` (configured to line length 88)
- Format notebooks: `nbqa black .`
- Lint: `ruff check pulse tests`
- Note: these commands are also enforced by pre-commit.

### Tests
- Run tests:
  ```bash
  make test
  # or directly
  pytest
  ```
- Many tests require OAuth credentials. Set:
  - `PULSE_CLIENT_ID`
  - `PULSE_CLIENT_SECRET`
  - Optional: `PULSE_TOKEN_URL`, `PULSE_AUDIENCE`
- CI runs pytest with:
  ```bash
  pytest -q --disable-warnings --maxfail=1 --vcr-record=none
  ```

### HTTP Cassette Recording (pytest-vcr)
- Re-record all cassettes from scratch:
  ```bash
  make vcr-record
  ```

### Packaging
```bash
python -m build
```

### Notes
- Keep changes backward compatible with existing models and APIs.
- Avoid committing large datasets or generated notebook outputs.

Feel free to open issues or submit pull requests at the [GitHub repo](https://github.com/researchwiseai/pulse-py).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
