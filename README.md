# pulse-sdk
Idiomatic, type-safe Python client for the Researchwise AI Pulse REST API.

## Features
- Low‑level CoreClient for direct API calls: embeddings, similarity, themes, sentiment
- High‑level Analyzer for orchestrating multi‑step workflows with caching
- Built-in processes: ThemeGeneration, ThemeAllocation, SentimentProcess, Cluster
- Result helpers: pandas DataFrame conversion, summaries, visualizations (bar charts, scatter, dendrogram)
- On‑disk and in‑memory caching via diskcache
- First-class interop with pandas, NumPy, and scikit‑learn

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
python -m venv .venv         # create a virtual environment (optional but recommended)
source .venv/bin/activate    # on Windows use `.venv\\Scripts\\activate`
pip install -e .[dev]        # install pulse-sdk plus dev tools (pytest, black, ruff, etc.)
```

## Getting Started

Once installed, you can quickly try out the core and DSL APIs.

### CoreClient example
### CoreClient
```python
from pulse.core.client import CoreClient

# Unauthenticated (dev) environment
client = CoreClient()  # default to dev environment
emb = client.create_embeddings(["Hello world", "Goodbye"])  # sync "fast" call
print(emb.embeddings)

# Submit a long-running job asynchronously
job = client.create_embeddings(["foo"] * 300, fast=False, await_job_result=False)
result = job.wait()
```

### CoreClient with Authentication

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

### Summarize text

```python
from pulse.starters import summarize

# Works with a list of strings or a file path
summary = summarize("reviews.txt", question="What do people think?")
print(summary.summary)
```

### Cluster texts

```python
from pulse.starters import cluster_analysis

# Cluster comments from a CSV file into two groups
clusters = cluster_analysis("reviews.csv", k=2)
print(clusters.clusters)
```

### Extract elements

```python
client = CoreClient()
resp = client.extract_elements(
    texts=["The food was great and the service was slow."],
    categories=["food", "service"],
    fast=True,
)
print(resp.extractions)
```

### Polling asynchronous jobs

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

### DSL Builder with Monitoring

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
Clone the project and install as above.  We recommend using a virtual environment.

1. Set up pre-commit (formats, linters, and tests on each commit). **Ensure your virtual environment is activated** so that `pre-commit` refers to your venv installation:
   ```bash
   # install or upgrade to a recent pre-commit
   pip install 'pre-commit>=2.9.2'
   # install git hook scripts
   pre-commit install
   # run all hooks against all files now
   pre-commit run --all-files
   ```
   If you still see an old version, invoke directly via Python:
   ```bash
   python -m pre_commit run --all-files
   ```
2. Run tests without re-recording cassettes:
   ```bash
   make test
   # or, directly:
   pytest --vcr-record=none
   ```
3. To reset and re-record all VCR cassettes from scratch:
   ```bash
   make vcr-record
   ```
4. Run code formatters & linters:
   ```bash
   black .                      # format Python source
   nbqa black .                 # format Jupyter notebooks
   ruff check pulse tests  # lint only code and tests
   ```
5. Build distributions:
   ```bash
   python -m build
   ```

Feel free to open issues or submit pull requests at the [GitHub repo](https://github.com/researchwiseai/pulse-py).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
