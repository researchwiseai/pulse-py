# Dependency Troubleshooting Guide

This guide helps you resolve common dependency conflicts and installation issues when working with the Pulse SDK.

## Understanding Dependency Conflicts

Dependency conflicts occur when different packages require incompatible versions of the same dependency. The Pulse SDK is designed to minimize conflicts by:

1. **Minimal core dependencies** - Only essential packages are required
2. **Optional feature sets** - Add only what you need
3. **Flexible version ranges** - Compatible with a wide range of versions
4. **Clear error messages** - Specific guidance for resolution

## Common Conflict Scenarios

### 1. Pydantic Version Conflicts

**Problem:**
```
ERROR: pulse-sdk 0.6.1 has requirement pydantic>=2.0, but you have pydantic 1.10.0
```

**Cause:** You have an older version of Pydantic installed, but Pulse SDK requires v2.0+

**Solutions:**

**Option A: Upgrade Pydantic (Recommended)**
```bash
pip install --upgrade pydantic
pip install pulse-sdk[all]
```

**Option B: Check what's using old Pydantic**
```bash
# Find packages that depend on pydantic
pip show pydantic
pipdeptree -p pydantic

# Upgrade conflicting packages
pip install --upgrade fastapi sqlalchemy-pydantic
```

**Option C: Fresh environment**
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install pulse-sdk[all]
```

### 2. NumPy Version Conflicts

**Problem:**
```
ERROR: pulse-sdk has requirement numpy>=1.21.0, but you have numpy 1.19.5
```

**Cause:** Older NumPy version incompatible with scikit-learn requirements

**Solutions:**

**Check NumPy dependents:**
```bash
pip show numpy
pipdeptree -p numpy
```

**Upgrade NumPy and dependents:**
```bash
pip install --upgrade numpy scipy pandas scikit-learn
pip install pulse-sdk[analysis]
```

**Use minimal installation if NumPy not needed:**
```bash
pip install pulse-sdk[minimal]
```

### 3. Scikit-learn Version Conflicts

**Problem:**
```
ERROR: pulse-sdk has requirement scikit-learn>=1.4, but you have scikit-learn 1.2.0
```

**Cause:** Older scikit-learn version lacks required features

**Solutions:**

**Upgrade scikit-learn:**
```bash
pip install --upgrade scikit-learn
pip install pulse-sdk[analysis]
```

**Skip analysis features if not needed:**
```bash
pip install pulse-sdk[minimal,visualization]
```

### 4. Multiple Package Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Cause:** Complex web of incompatible package versions

**Solutions:**

**Option A: Fresh Virtual Environment (Most Reliable)**
```bash
# Deactivate current environment
deactivate

# Create new environment
python -m venv pulse_clean
source pulse_clean/bin/activate  # Windows: pulse_clean\Scripts\activate

# Install with latest pip
pip install --upgrade pip setuptools wheel
pip install pulse-sdk[all]
```

**Option B: Incremental Installation**
```bash
# Start with minimal
pip install pulse-sdk[minimal]

# Test basic functionality
python -c "from pulse.core.client import CoreClient; print('✅ Core works')"

# Add features one by one
pip install pulse-sdk[analysis]
pip install pulse-sdk[visualization]
pip install pulse-sdk[caching]
```

**Option C: Force Reinstall**
```bash
pip install --force-reinstall --no-deps pulse-sdk
pip install --upgrade httpx pydantic typing-extensions
```

## Platform-Specific Issues

### macOS with Apple Silicon

**Problem:**
```
ERROR: Failed building wheel for numpy
```

**Solutions:**

**Use conda for scientific packages:**
```bash
conda install numpy pandas scikit-learn matplotlib
pip install pulse-sdk[minimal]
```

**Or upgrade build tools:**
```bash
pip install --upgrade pip setuptools wheel
export ARCHFLAGS="-arch arm64"
pip install pulse-sdk[all]
```

### Windows with Python from Microsoft Store

**Problem:**
```
ERROR: Microsoft Visual C++ 14.0 is required
```

**Solutions:**

**Use pre-compiled wheels:**
```bash
pip install --only-binary=all pulse-sdk[all]
```

**Or install Visual Studio Build Tools:**
1. Download Visual Studio Build Tools
2. Install C++ build tools
3. Restart terminal and retry installation

### Linux with System Python

**Problem:**
```
ERROR: Failed to build package, missing Python.h
```

**Solutions:**

**Install development headers:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# CentOS/RHEL/Fedora
sudo yum install python3-devel python3-pip
# or
sudo dnf install python3-devel python3-pip

# Then install Pulse SDK
pip install pulse-sdk[all]
```

## Version Compatibility Matrix

### Python Version Compatibility

| Python Version | Pulse SDK | NumPy | Pandas | Scikit-learn | Notes |
|----------------|-----------|-------|--------|--------------|-------|
| 3.8 | ✅ All versions | ≥1.21.0 | ≥1.3.0 | ≥1.4 | Minimum supported |
| 3.9 | ✅ All versions | ≥1.21.0 | ≥1.3.0 | ≥1.4 | Recommended |
| 3.10 | ✅ All versions | ≥1.21.0 | ≥1.3.0 | ≥1.4 | Excellent performance |
| 3.11 | ✅ All versions | ≥1.21.0 | ≥1.3.0 | ≥1.4 | Best performance |
| 3.12 | ✅ All versions | ≥1.21.0 | ≥1.3.0 | ≥1.4 | Latest features |

### Package Manager Compatibility

| Package Manager | Installation Command | Notes |
|-----------------|---------------------|-------|
| pip | `pip install pulse-sdk[all]` | Recommended, best support |
| pipenv | `pipenv install pulse-sdk[all]` | Good, handles virtual envs |
| poetry | `poetry add pulse-sdk[all]` | Good, modern dependency management |
| conda | `pip install pulse-sdk[all]` | Install via pip in conda env |
| uv | `uv pip install pulse-sdk[all]` | Fastest installation |

## Diagnostic Commands

### Check Current Environment

```bash
# Python and pip versions
python --version
pip --version

# Installed packages
pip list | grep -E "(pulse|numpy|pandas|pydantic|scikit-learn)"

# Check for conflicts
pip check

# Dependency tree
pip install pipdeptree
pipdeptree -p pulse-sdk
```

### Test Installation

```bash
# Basic import test
python -c "import pulse; print(f'Pulse SDK {pulse.__version__} installed')"

# Feature availability test
python -c "
try:
    from pulse.core.client import CoreClient
    print('✅ Core client available')
except ImportError as e:
    print(f'❌ Core client failed: {e}')

try:
    import numpy, pandas, sklearn
    print('✅ Analysis features available')
except ImportError as e:
    print(f'❌ Analysis features failed: {e}')

try:
    import matplotlib, seaborn
    print('✅ Visualization features available')
except ImportError as e:
    print(f'❌ Visualization features failed: {e}')
"
```

### API Connection Test

```bash
python -c "
from pulse.core.client import CoreClient
try:
    client = CoreClient()
    result = client.analyze_sentiment(['test'], fast=True)
    print('✅ API connection successful')
except Exception as e:
    print(f'❌ API connection failed: {e}')
"
```

## Resolution Strategies

### Strategy 1: Clean Slate Approach

Best for complex conflicts:

```bash
# 1. Create fresh environment
python -m venv pulse_env
source pulse_env/bin/activate

# 2. Upgrade pip and tools
pip install --upgrade pip setuptools wheel

# 3. Install Pulse SDK
pip install pulse-sdk[all]

# 4. Verify installation
python -c "import pulse; print('Success!')"
```

### Strategy 2: Minimal First Approach

Best for cautious installation:

```bash
# 1. Install minimal version
pip install pulse-sdk[minimal]

# 2. Test core functionality
python -c "from pulse.core.client import CoreClient; print('Core OK')"

# 3. Add features incrementally
pip install pulse-sdk[analysis]
pip install pulse-sdk[visualization]

# 4. Test each addition
python -c "import numpy, pandas; print('Analysis OK')"
```

### Strategy 3: Constraint-Based Approach

Best for specific version requirements:

```bash
# 1. Create constraints file
cat > constraints.txt << EOF
numpy>=1.21.0,<2.0
pandas>=1.3.0,<3.0
scikit-learn>=1.4,<2.0
pydantic>=2.0,<3.0
EOF

# 2. Install with constraints
pip install -c constraints.txt pulse-sdk[all]
```

### Strategy 4: Conda + Pip Hybrid

Best for scientific computing environments:

```bash
# 1. Install scientific packages with conda
conda install numpy pandas scikit-learn matplotlib seaborn

# 2. Install Pulse SDK with pip
pip install pulse-sdk[minimal]

# 3. Verify all features work
python -c "from pulse.analysis.analyzer import Analyzer; print('All features OK')"
```

## Getting Help

If you're still experiencing issues after trying these solutions:

### 1. Gather Information

```bash
# System information
python --version
pip --version
uname -a  # Linux/macOS
systeminfo  # Windows

# Package information
pip list > installed_packages.txt
pip check > conflicts.txt 2>&1
```

### 2. Create Minimal Reproduction

```bash
# Fresh environment
python -m venv test_env
source test_env/bin/activate
pip install --upgrade pip

# Try installation
pip install pulse-sdk[all] > install_log.txt 2>&1
```

### 3. Report the Issue

Create a GitHub issue with:
- Operating system and Python version
- Complete error message and traceback
- Output from diagnostic commands
- Installation method attempted
- Any custom requirements or constraints

**GitHub Issues:** https://github.com/researchwiseai/pulse-py/issues

### 4. Community Support

- **Stack Overflow:** Tag questions with `pulse-sdk` and `python`
- **GitHub Discussions:** For general questions and community help

## Prevention Tips

### 1. Use Virtual Environments

Always use virtual environments to avoid conflicts:

```bash
# Create project-specific environment
python -m venv myproject_env
source myproject_env/bin/activate
pip install pulse-sdk[all]
```

### 2. Pin Dependencies in Production

Create a requirements.txt with exact versions:

```bash
# Generate from working environment
pip freeze > requirements.txt

# Install exact versions elsewhere
pip install -r requirements.txt
```

### 3. Regular Updates

Keep dependencies updated to avoid security issues:

```bash
# Check for outdated packages
pip list --outdated

# Update specific packages
pip install --upgrade pulse-sdk numpy pandas

# Or update all
pip install --upgrade $(pip list --outdated --format=freeze | cut -d= -f1)
```

### 4. Use Dependency Scanning

Monitor for vulnerabilities:

```bash
# Install scanning tools
pip install pip-audit safety

# Scan for vulnerabilities
pip-audit
safety check
```

By following these guidelines and using the appropriate resolution strategy for your situation, you should be able to resolve most dependency conflicts and get the Pulse SDK working smoothly in your environment.
