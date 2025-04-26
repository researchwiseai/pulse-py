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

client = CoreClient()  # default to dev environment
emb = client.create_embeddings(["Hello world", "Goodbye"])  # sync "fast" call
print(emb.embeddings)
```

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

## Examples
You can find Jupyter notebooks demonstrating both the high-level and DSL APIs under the `examples/` directory:
```bash
jupyter notebook examples/high_level_api.ipynb
jupyter notebook examples/dsl_api.ipynb
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
