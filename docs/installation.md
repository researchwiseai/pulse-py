# Installation Guide

This guide provides comprehensive installation options for the Pulse SDK, from minimal core functionality to full-featured installations with all optional dependencies.

## Quick Installation

For most users, the recommended installation includes all features:

```bash
pip install pulse-sdk[all]
```

This installs the SDK with all optional dependencies for the complete experience.

## Installation Options

### 1. Minimal Core Installation

Install only the essential dependencies for basic API access:

```bash
pip install pulse-sdk[minimal]
```

**Includes:**
- `httpx` - HTTP client for API requests
- `pydantic` - Data validation and serialization
- `typing-extensions` - Type hints for older Python versions

**Use when:**
- You only need basic API calls (embeddings, sentiment, themes)
- Working in resource-constrained environments
- Building microservices with minimal dependencies
- You'll handle data processing with your own tools

**Example usage:**
```python
from pulse.core.client import CoreClient

client = CoreClient()
result = client.analyze_sentiment(["Hello world"])
print(result.results[0].sentiment)
```

### 2. Analysis Features

Add data science and machine learning capabilities:

```bash
pip install pulse-sdk[analysis]
```

**Adds:**
- `numpy` - Numerical computing for embeddings and vectors
- `pandas` - Data manipulation and DataFrame support
- `scikit-learn` - Machine learning utilities for clustering

**Use when:**
- Working with embeddings and similarity calculations
- Need DataFrame conversion of results
- Using clustering algorithms
- Performing numerical analysis on text data

**Example usage:**
```python
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import ThemeGeneration

analyzer = Analyzer(dataset=texts, processes=[ThemeGeneration()])
results = analyzer.run()
df = results.theme_generation.to_dataframe()  # Requires pandas
```

### 3. Visualization Features

Add plotting and chart generation:

```bash
pip install pulse-sdk[visualization]
```

**Adds:**
- `matplotlib` - Basic plotting and charts
- `seaborn` - Statistical visualizations and themes

**Use when:**
- Creating charts from analysis results
- Generating reports with visualizations
- Exploring data interactively in Jupyter notebooks

**Example usage:**
```python
from pulse.analysis.results import SentimentResults

results = SentimentResults(...)
results.plot_distribution()  # Requires matplotlib/seaborn
```

### 4. Natural Language Processing

Add advanced text processing utilities:

```bash
pip install pulse-sdk[nlp]
```

**Adds:**
- `textblob` - Natural language processing utilities

**Use when:**
- Need additional text preprocessing
- Working with linguistic features
- Performing text normalization

### 5. Caching and Performance

Add caching capabilities for improved performance:

```bash
pip install pulse-sdk[caching]
```

**Adds:**
- `diskcache` - On-disk caching for analysis results

**Use when:**
- Running repeated analyses on the same data
- Working with large datasets
- Need persistent caching across sessions

**Example usage:**
```python
from pulse.analysis.analyzer import Analyzer

# Results are automatically cached to disk
with Analyzer(cache_dir=".pulse_cache") as analyzer:
    results = analyzer.run()  # Cached for future runs
```

### 6. Progress Tracking

Add progress bars for long-running operations:

```bash
pip install pulse-sdk[progress]
```

**Adds:**
- `tqdm` - Progress bars for long-running operations

**Use when:**
- Processing large datasets
- Need visual feedback on operation progress
- Running batch operations

### 7. Combined Installations

Install multiple feature sets together:

```bash
# Data science workflow
pip install pulse-sdk[analysis,visualization,caching]

# Complete NLP pipeline
pip install pulse-sdk[analysis,nlp,progress]

# Development setup
pip install pulse-sdk[all,dev]
```

### 8. Development Installation

For contributors and advanced users:

```bash
pip install pulse-sdk[dev]
```

**Includes all testing, formatting, and documentation tools:**
- `pytest` - Testing framework
- `black` - Code formatting
- `ruff` - Fast Python linter
- `pre-commit` - Git hooks for code quality
- `bandit` - Security vulnerability scanner
- `pip-audit` - Dependency vulnerability scanner
- `mkdocs` - Documentation generator

## Installation Commands by Use Case

### Data Scientists and Researchers
```bash
# Complete data science stack
pip install pulse-sdk[analysis,visualization,caching,progress]
```

### Web Developers and API Integration
```bash
# Lightweight for web services
pip install pulse-sdk[minimal,progress]
```

### Content Analysis and NLP
```bash
# Full text processing capabilities
pip install pulse-sdk[analysis,nlp,visualization]
```

### Production Deployments
```bash
# Optimized for production with caching
pip install pulse-sdk[analysis,caching]
```

### Interactive Development and Notebooks
```bash
# Full-featured for exploration
pip install pulse-sdk[all]
```

## Version Compatibility

### Python Version Support

The Pulse SDK supports Python 3.8 and later:

- **Python 3.8+**: All features supported
- **Python 3.9+**: Recommended for best performance
- **Python 3.10+**: Latest type hints and optimizations
- **Python 3.11+**: Fastest performance
- **Python 3.12+**: Latest features and improvements

### Dependency Version Compatibility

| Dependency | Minimum Version | Recommended | Notes |
|------------|----------------|-------------|-------|
| httpx | 0.24.0 | Latest | HTTP client for API requests |
| pydantic | 2.0 | Latest | Data validation, v2+ required |
| numpy | 1.21.0 | Latest | Numerical computing |
| pandas | 1.3.0 | Latest | Data manipulation |
| scikit-learn | 1.4 | Latest | Machine learning utilities |
| matplotlib | 3.5.0 | Latest | Plotting and visualization |
| seaborn | 0.11.0 | Latest | Statistical visualizations |

### Package Manager Compatibility

The Pulse SDK works with all major Python package managers:

**pip (recommended)**
```bash
pip install pulse-sdk[all]
```

**pipenv**
```bash
pipenv install pulse-sdk[all]
```

**poetry**
```bash
poetry add pulse-sdk[all]
```

**conda**
```bash
# Install from PyPI via pip in conda environment
conda install pip
pip install pulse-sdk[all]
```

**uv (fastest)**
```bash
uv pip install pulse-sdk[all]
```

## Troubleshooting Installation Issues

### Common Problems and Solutions

#### 1. Dependency Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**

**Option A: Fresh Virtual Environment (Recommended)**
```bash
# Create clean environment
python -m venv pulse_env
source pulse_env/bin/activate  # Windows: pulse_env\Scripts\activate
pip install --upgrade pip
pip install pulse-sdk[all]
```

**Option B: Upgrade Conflicting Packages**
```bash
# Check for conflicts
pip check

# Upgrade specific packages
pip install --upgrade numpy pandas scikit-learn

# Then install Pulse SDK
pip install pulse-sdk[all]
```

**Option C: Use Minimal Installation**
```bash
# Install minimal version first
pip install pulse-sdk[minimal]

# Add features incrementally
pip install pulse-sdk[analysis]
pip install pulse-sdk[visualization]
```

#### 2. Network and Timeout Issues

**Problem:**
```
ERROR: Could not install packages due to an EnvironmentError: HTTPSConnectionPool
```

**Solutions:**

**Increase timeout:**
```bash
pip install --timeout 300 pulse-sdk[all]
```

**Use different index:**
```bash
pip install -i https://pypi.org/simple/ pulse-sdk[all]
```

**Install with retries:**
```bash
pip install --retries 5 pulse-sdk[all]
```

#### 3. Permission Issues

**Problem:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solutions:**

**Use virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate
pip install pulse-sdk[all]
```

**User installation:**
```bash
pip install --user pulse-sdk[all]
```

#### 4. Platform-Specific Issues

**macOS with Apple Silicon:**
```bash
# Ensure compatible versions
pip install --upgrade pip setuptools wheel
pip install pulse-sdk[all]
```

**Windows with Python from Microsoft Store:**
```bash
# Use py launcher
py -m pip install pulse-sdk[all]
```

**Linux with system Python:**
```bash
# Install development headers if needed
sudo apt-get install python3-dev  # Ubuntu/Debian
# or
sudo yum install python3-devel    # CentOS/RHEL

pip install pulse-sdk[all]
```

#### 5. Version Conflicts with Existing Packages

**Problem:**
```
ERROR: pulse-sdk 0.6.1 has requirement pydantic>=2.0, but you have pydantic 1.10.0
```

**Solutions:**

**Check current versions:**
```bash
pip list | grep -E "(pydantic|numpy|pandas|scikit-learn)"
```

**Upgrade conflicting packages:**
```bash
pip install --upgrade pydantic numpy pandas scikit-learn
pip install pulse-sdk[all]
```

**Use version constraints:**
```bash
pip install "pydantic>=2.0" "numpy>=1.21.0"
pip install pulse-sdk[all]
```

### Verification Steps

After installation, verify everything works:

**1. Basic Import Test:**
```python
try:
    import pulse
    print("✅ Pulse SDK imported successfully")
    print(f"Version: {pulse.__version__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

**2. Feature Availability Test:**
```python
# Test core functionality
try:
    from pulse.core.client import CoreClient
    print("✅ Core client available")
except ImportError:
    print("❌ Core client not available")

# Test analysis features
try:
    import numpy, pandas, sklearn
    from pulse.analysis.analyzer import Analyzer
    print("✅ Analysis features available")
except ImportError:
    print("❌ Analysis features not available - install with [analysis]")

# Test visualization features
try:
    import matplotlib, seaborn
    print("✅ Visualization features available")
except ImportError:
    print("❌ Visualization features not available - install with [visualization]")
```

**3. API Connection Test:**
```python
from pulse.core.client import CoreClient

try:
    client = CoreClient()
    # Test with minimal request
    result = client.analyze_sentiment(["test"], fast=True)
    print("✅ API connection successful")
except Exception as e:
    print(f"❌ API connection failed: {e}")
```

### Getting Help

If you continue to experience installation issues:

1. **Check system requirements:**
   - Python 3.8 or later
   - pip 21.0 or later
   - Internet connection for package downloads

2. **Gather diagnostic information:**
   ```bash
   python --version
   pip --version
   pip list | grep pulse
   ```

3. **Create a minimal reproduction:**
   - Fresh virtual environment
   - Minimal installation command
   - Error message and traceback

4. **Report the issue:**
   - [GitHub Issues](https://github.com/researchwiseai/pulse-py/issues)
   - Include Python version, OS, and complete error message
   - Mention which installation method you tried

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Get up and running in 5 minutes
2. **[Authentication Setup](authentication.md)** - Configure API access
3. **[Core Client Guide](core-client.md)** - Learn the basic API
4. **[Examples](https://github.com/researchwiseai/pulse-py/tree/main/examples)** - Jupyter notebooks with detailed examples

Choose your installation based on your needs, and you'll be analyzing text with the Pulse SDK in no time!
