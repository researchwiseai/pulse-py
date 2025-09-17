# Project Structure

## Package Organization

The Pulse SDK follows a layered architecture with clear separation of concerns:

```
pulse/                          # Main package
├── __init__.py                 # Package version and exports
├── auth.py                     # OAuth2 authentication classes
├── config.py                   # Environment-based configuration
├── dsl.py                      # DSL builder for custom workflows
├── starters.py                 # High-level convenience functions
├── core/                       # Low-level API client
│   ├── client.py              # CoreClient for direct API calls
│   ├── models.py              # Pydantic response models
│   ├── jobs.py                # Async job handling
│   ├── batching.py            # Request batching utilities
│   ├── retry.py               # Retry logic for HTTP requests
│   ├── exceptions.py          # Custom exception classes
│   ├── gzip_client.py         # Compressed request handling
│   └── utils.py               # Utility functions (chunking, etc.)
└── analysis/                   # High-level analysis workflows
    ├── analyzer.py            # Analyzer orchestrator class
    ├── processes.py           # Built-in analysis processes
    └── results.py             # Result handling and visualization
```

## Key Directories

### `/docs/`
Documentation source files for MkDocs:
- `index.md`: Main documentation entry point
- Individual feature documentation (analyzer.md, dsl.md, etc.)

### `/examples/`
Jupyter notebooks demonstrating SDK usage:
- `high_level_api.ipynb`: Analyzer and starters examples
- `dsl_api.ipynb`: DSL builder examples
- `low_level_api.ipynb`: CoreClient examples

### `/tests/`
Comprehensive test suite:
- `test_*.py`: Feature-specific test files
- `cassettes/`: HTTP request/response recordings (pytest-vcr)
- `fixtures/`: Test data files

### `/site/`
Generated documentation (MkDocs build output) - not version controlled in development

## Architecture Layers

### 1. Core Layer (`pulse.core`)
- **Purpose**: Direct API communication
- **Key Classes**: `CoreClient`, Pydantic models
- **Usage**: When you need fine-grained control over API calls

### 2. Analysis Layer (`pulse.analysis`)
- **Purpose**: Multi-step workflows with caching
- **Key Classes**: `Analyzer`, process classes
- **Usage**: For complex analysis pipelines with intermediate caching

### 3. DSL Layer (`pulse.dsl`)
- **Purpose**: Declarative workflow building
- **Key Classes**: `Workflow`
- **Usage**: For building reusable, configurable analysis pipelines

### 4. Starters Layer (`pulse.starters`)
- **Purpose**: Simple, one-line functions for common tasks
- **Usage**: Quick analysis without configuration complexity

## File Naming Conventions

- **Snake case** for all Python files and directories
- **Test files** prefixed with `test_`
- **Cassette files** match test function names
- **Documentation files** use descriptive names matching features

## Import Patterns

- Core functionality: `from pulse.core.client import CoreClient`
- High-level analysis: `from pulse.analysis.analyzer import Analyzer`
- DSL building: `from pulse.dsl import Workflow`
- Convenience functions: `from pulse.starters import summarize`
- Authentication: `from pulse.auth import ClientCredentialsAuth`