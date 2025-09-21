# Technology Stack

## Core Technologies

- **Python**: 3.8+ required
- **HTTP Client**: httpx for async/sync HTTP requests
- **Data Validation**: Pydantic v2+ for type-safe models
- **Data Science**: NumPy, pandas, scikit-learn (>=1.4), matplotlib, seaborn
- **Caching**: diskcache for persistent caching
- **Text Processing**: textblob for NLP utilities
- **Progress**: tqdm for progress bars

## Development Tools

- **Testing**: pytest with pytest-mock, pytest-vcr for HTTP recording
- **Code Formatting**: black (line length 88, Python 3.8+ target)
- **Linting**: ruff with E501 line length checks
- **Notebook Formatting**: nbqa for Jupyter notebook code formatting
- **Pre-commit**: Automated formatting and linting on commit
- **Documentation**: MkDocs with mkdocs-material theme

## Build System

- **Build Backend**: setuptools with pyproject.toml configuration
- **Package Manager**: pip with virtual environments recommended

## Common Commands

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
make test
# or
pytest

# Re-record HTTP cassettes
make vcr-record

# Clean VCR cassettes
make vcr-clean
```

### Code Quality
```bash
# Format Python code
black .

# Format notebooks
nbqa black .

# Lint code
ruff check pulse tests

# Run pre-commit on all files
pre-commit run --all-files
```

### Documentation
```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Serve docs locally
mkdocs serve

# Build static docs
mkdocs build
```

### Packaging
```bash
# Build distribution packages
python -m build
```

## Environment Configuration

The SDK uses production endpoints by default. All configuration can be overridden via environment variables:

Key environment variables:
- `PULSE_CLIENT_ID`, `PULSE_CLIENT_SECRET`: OAuth2 credentials
- `PULSE_BASE_URL`: API base URL (default: `https://pulse.researchwiseai.com/v1`)
- `PULSE_AUDIENCE`: OAuth2 audience (default: `https://core.researchwiseai.com/pulse/v1`)
- `PULSE_TOKEN_URL`, `PULSE_AUTH_DOMAIN`: Auth configuration
