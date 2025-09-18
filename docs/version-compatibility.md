# Version Compatibility Guide

This guide provides comprehensive information about version compatibility for the Pulse SDK, including Python versions, dependency versions, and package manager support.

## Python Version Support

### Supported Python Versions

The Pulse SDK supports Python 3.8 and later with the following compatibility matrix:

| Python Version | Support Status | Performance | Recommended Use |
|----------------|----------------|-------------|-----------------|
| **3.8** | ✅ Full Support | Good | Minimum supported version |
| **3.9** | ✅ Full Support | Better | Stable production use |
| **3.10** | ✅ Full Support | Excellent | Recommended for new projects |
| **3.11** | ✅ Full Support | Best | Optimal performance |
| **3.12** | ✅ Full Support | Best | Latest features and optimizations |

### Python Version Features

**Python 3.8+**
- All Pulse SDK features supported
- Type hints with `typing-extensions`
- Async/await support for job polling

**Python 3.9+**
- Improved type hint performance
- Better dictionary merge operations
- Enhanced error messages

**Python 3.10+**
- Structural pattern matching (used internally)
- Better error locations in tracebacks
- Performance improvements

**Python 3.11+**
- Significant performance improvements (10-60% faster)
- Better error messages
- Faster startup time

**Python 3.12+**
- Latest performance optimizations
- Improved memory usage
- Enhanced debugging capabilities

## Core Dependencies

### Required Dependencies (Always Installed)

| Package | Minimum Version | Latest Tested | Purpose | Notes |
|---------|----------------|---------------|---------|-------|
| **httpx** | 0.24.0 | 0.27.0 | HTTP client | Async/sync API requests |
| **pydantic** | 2.0.0 | 2.8.0 | Data validation | V2+ required for performance |
| **typing-extensions** | 4.0.0 | 4.12.0 | Type hints | Backport for older Python |

### Version Compatibility Notes

**httpx**
- Minimum 0.24.0 required for timeout handling
- 0.25.0+ recommended for better error messages
- 0.27.0+ includes performance improvements

**pydantic**
- V2.0+ required (breaking change from V1)
- V2.4+ recommended for better performance
- V2.6+ includes improved error messages

**typing-extensions**
- Required for Python < 3.10
- Provides backports of newer typing features
- Safe to upgrade to latest version

## Optional Dependencies

### Analysis Features (`pulse-sdk[analysis]`)

| Package | Minimum Version | Latest Tested | Purpose | Compatibility Notes |
|---------|----------------|---------------|---------|-------------------|
| **numpy** | 1.21.0 | 1.26.0 | Numerical computing | Python 3.8+ support |
| **pandas** | 1.3.0 | 2.2.0 | Data manipulation | Requires numpy >= 1.21.0 |
| **scikit-learn** | 1.4.0 | 1.4.0 | Machine learning | Requires numpy >= 1.21.0 |

**Version Compatibility Matrix:**

| Python | NumPy | Pandas | Scikit-learn | Status |
|--------|-------|--------|--------------|--------|
| 3.8 | 1.21.0+ | 1.3.0+ | 1.4.0+ | ✅ Supported |
| 3.9 | 1.21.0+ | 1.3.0+ | 1.4.0+ | ✅ Supported |
| 3.10 | 1.21.0+ | 1.3.0+ | 1.4.0+ | ✅ Recommended |
| 3.11 | 1.21.0+ | 1.3.0+ | 1.4.0+ | ✅ Recommended |
| 3.12 | 1.21.0+ | 1.3.0+ | 1.4.0+ | ✅ Latest |

### Visualization Features (`pulse-sdk[visualization]`)

| Package | Minimum Version | Latest Tested | Purpose | Compatibility Notes |
|---------|----------------|---------------|---------|-------------------|
| **matplotlib** | 3.5.0 | 3.8.0 | Plotting | Requires numpy >= 1.21.0 |
| **seaborn** | 0.11.0 | 0.13.0 | Statistical plots | Requires matplotlib >= 3.5.0 |

### Additional Features

| Feature Set | Package | Minimum Version | Latest Tested | Notes |
|-------------|---------|----------------|---------------|-------|
| **NLP** | textblob | 0.17.0 | 0.18.0 | Text processing utilities |
| **Caching** | diskcache | 5.4.0 | 5.6.0 | On-disk caching |
| **Progress** | tqdm | 4.64.0 | 4.66.0 | Progress bars |

## Package Manager Compatibility

### pip (Recommended)

**Supported pip versions:** 21.0+

```bash
# Standard installation
pip install pulse-sdk[all]

# With version constraints
pip install "pulse-sdk[all]>=0.6.0,<1.0.0"

# Upgrade existing installation
pip install --upgrade pulse-sdk[all]
```

**pip Features Used:**
- PEP 621 metadata support
- Optional dependencies (extras)
- Version constraints
- Dependency resolution

### pipenv

**Supported pipenv versions:** 2022.1.8+

```bash
# Add to Pipfile
pipenv install pulse-sdk[all]

# With version constraints
pipenv install "pulse-sdk[all]>=0.6.0"

# Development dependencies
pipenv install pulse-sdk[dev] --dev
```

**Pipfile example:**
```toml
[packages]
pulse-sdk = {extras = ["all"], version = ">=0.6.0"}

[dev-packages]
pulse-sdk = {extras = ["dev"], version = "*"}
```

### poetry

**Supported poetry versions:** 1.2.0+

```bash
# Add dependency
poetry add pulse-sdk[all]

# With version constraints
poetry add "pulse-sdk[all]>=0.6.0,<1.0.0"

# Development dependencies
poetry add --group dev pulse-sdk[dev]
```

**pyproject.toml example:**
```toml
[tool.poetry.dependencies]
python = "^3.8"
pulse-sdk = {extras = ["all"], version = "^0.6.0"}

[tool.poetry.group.dev.dependencies]
pulse-sdk = {extras = ["dev"], version = "^0.6.0"}
```

### conda

**Supported conda versions:** 4.10.0+

```bash
# Create environment with pip
conda create -n pulse python=3.11
conda activate pulse

# Install via pip (recommended)
pip install pulse-sdk[all]

# Or install scientific packages via conda first
conda install numpy pandas scikit-learn matplotlib seaborn
pip install pulse-sdk[minimal]
```

**environment.yml example:**
```yaml
name: pulse-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scikit-learn>=1.4.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - pip
  - pip:
    - pulse-sdk[minimal]
```

### uv (Fastest)

**Supported uv versions:** 0.1.0+

```bash
# Install with uv (fastest)
uv pip install pulse-sdk[all]

# Create virtual environment
uv venv
source .venv/bin/activate
uv pip install pulse-sdk[all]
```

## Platform Compatibility

### Operating Systems

| Platform | Support Status | Notes |
|----------|----------------|-------|
| **Linux** | ✅ Full Support | All distributions |
| **macOS** | ✅ Full Support | Intel and Apple Silicon |
| **Windows** | ✅ Full Support | Windows 10+ |

### Architecture Support

| Architecture | Support Status | Notes |
|--------------|----------------|-------|
| **x86_64** | ✅ Full Support | Standard 64-bit |
| **ARM64** | ✅ Full Support | Apple Silicon, ARM servers |
| **x86** | ⚠️ Limited | 32-bit, not recommended |

### Platform-Specific Notes

**macOS with Apple Silicon (M1/M2/M3)**
- All packages have native ARM64 wheels
- No Rosetta emulation required
- Excellent performance

**Windows**
- Pre-compiled wheels available for all dependencies
- No Visual Studio Build Tools required
- Works with Python from python.org or Microsoft Store

**Linux**
- Works on all major distributions
- May require development headers for source builds
- Excellent performance on all architectures

## Version Upgrade Guide

### Upgrading Pulse SDK

**From 0.5.x to 0.6.x:**
```bash
# Check current version
pip show pulse-sdk

# Upgrade to latest
pip install --upgrade pulse-sdk[all]

# Verify upgrade
python -c "import pulse; print(pulse.__version__)"
```

**Breaking Changes:**
- None in 0.6.x series
- Backward compatible with 0.5.x

### Upgrading Dependencies

**Safe Upgrade Strategy:**
```bash
# Check current versions
pip list | grep -E "(numpy|pandas|scikit-learn|pydantic)"

# Upgrade incrementally
pip install --upgrade pydantic
pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade scikit-learn

# Test after each upgrade
python -c "from pulse.core.client import CoreClient; print('OK')"
```

**Bulk Upgrade (More Risky):**
```bash
# Upgrade all at once
pip install --upgrade pulse-sdk[all] numpy pandas scikit-learn pydantic

# Test thoroughly
python -c "
from pulse.core.client import CoreClient
from pulse.analysis.analyzer import Analyzer
print('All components working')
"
```

## Compatibility Testing

### Test Your Environment

```bash
# Create test script
cat > test_compatibility.py << 'EOF'
import sys
import importlib.util

def test_python_version():
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} not supported (need 3.8+)")
        return False

def test_package(name, min_version=None):
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            print(f"❌ {name} not installed")
            return False

        module = importlib.import_module(name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {name} {version} available")
        return True
    except ImportError as e:
        print(f"❌ {name} import failed: {e}")
        return False

def main():
    print("Pulse SDK Compatibility Test")
    print("=" * 40)

    # Test Python version
    python_ok = test_python_version()

    # Test core dependencies
    print("\nCore Dependencies:")
    core_ok = all([
        test_package('httpx'),
        test_package('pydantic'),
        test_package('typing_extensions'),
    ])

    # Test optional dependencies
    print("\nOptional Dependencies:")
    test_package('numpy')
    test_package('pandas')
    test_package('sklearn')
    test_package('matplotlib')
    test_package('seaborn')
    test_package('textblob')
    test_package('diskcache')
    test_package('tqdm')

    # Test Pulse SDK
    print("\nPulse SDK:")
    pulse_ok = test_package('pulse')

    if python_ok and core_ok and pulse_ok:
        print("\n✅ Environment is compatible with Pulse SDK")
    else:
        print("\n❌ Environment has compatibility issues")

if __name__ == '__main__':
    main()
EOF

# Run compatibility test
python test_compatibility.py
```

### Continuous Integration Testing

**GitHub Actions example:**
```yaml
name: Compatibility Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pulse-sdk[all]

    - name: Test compatibility
      run: |
        python -c "
        import pulse
        from pulse.core.client import CoreClient
        from pulse.analysis.analyzer import Analyzer
        print(f'✅ Pulse SDK {pulse.__version__} working on Python {sys.version}')
        "
```

## Troubleshooting Version Issues

### Common Version Conflicts

**Problem:** Incompatible dependency versions
**Solution:** See [Dependency Troubleshooting Guide](dependency-troubleshooting.md)

**Problem:** Python version too old
**Solution:** Upgrade Python or use older Pulse SDK version

**Problem:** Package manager doesn't support extras
**Solution:** Install dependencies manually:
```bash
pip install httpx pydantic typing-extensions
pip install numpy pandas scikit-learn  # for analysis
pip install matplotlib seaborn          # for visualization
pip install pulse-sdk
```

### Version Pinning for Production

**requirements.txt example:**
```txt
# Core Pulse SDK
pulse-sdk[all]==0.6.1

# Pin critical dependencies
httpx==0.27.0
pydantic==2.8.0
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.0
seaborn==0.13.0
```

**Dockerfile example:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify installation
RUN python -c "import pulse; print(f'Pulse SDK {pulse.__version__} ready')"
```

## Getting Help

If you encounter version compatibility issues:

1. **Check this guide** for known compatibility information
2. **Run the compatibility test** script above
3. **Check our troubleshooting guide** for common solutions
4. **Report issues** on GitHub with your environment details

**Environment Information to Include:**
```bash
python --version
pip --version
pip list | grep pulse
uname -a  # Linux/macOS
systeminfo  # Windows
```

The Pulse SDK is designed to work across a wide range of Python versions and environments. By following this compatibility guide, you should be able to find the right combination of versions for your specific use case.
