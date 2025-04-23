# pulse-client
Idiomatic, type-safe Python client for the Researchwise AI Pulse REST API.

## Features
- Low‑level CoreClient for direct API calls: embeddings, similarity, themes, sentiment
- High‑level Analyzer for orchestrating multi‑step workflows with caching
- Built-in processes: ThemeGeneration, ThemeAllocation, SentimentProcess, Cluster
- Result helpers: pandas DataFrame conversion, summaries, visualizations (bar charts, scatter, dendrogram)
- On‑disk and in‑memory caching via diskcache
- First-class interop with pandas, NumPy, and scikit‑learn

## Installation
Install from PyPI:
```bash
pip install pulse-client
```

For development:
```bash
git clone https://github.com/your-org/pulse-py.git
cd pulse-py
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart
### CoreClient
```python
from pulse_client.core.client import CoreClient

client = CoreClient()  # default to dev environment
emb = client.create_embeddings(["Hello world", "Goodbye"])  # sync "fast" call
print(emb.embeddings)
```

### Analyzer
```python
from pulse_client.analysis.analyzer import Analyzer
from pulse_client.analysis.processes import ThemeGeneration, SentimentProcess

texts = ["I love pizza", "I hate rain"]
processes = [ThemeGeneration(min_themes=2), SentimentProcess()]
with Analyzer(dataset=texts, processes=processes, cache_dir=".pulse_cache") as az:
    results = az.run()

print(results.theme_generation.to_dataframe())
print(results.sentiment.summary())
```

## Development
- Run tests: `pytest`
- Lint code: `black . && ruff .`
- Pre-commit hooks: `pre-commit run --all-files`

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
