 # Development Scratchpad

 ## Plan
- ✅ Initialize project skeleton:
  - ✅ create package directories
  - ✅ create pyproject.toml, .gitignore, dev tooling config
- ✅ Implement CoreClient sync methods for embeddings, similarity, themes, sentiment
- ✅ Implement Job model with refresh and wait
- ✅ Implement Pydantic models in core/models.py
- ✅ Implement config.py with base_url, timeouts
- ✅ Implement exceptions in core/exceptions.py
- ✅ Add basic tests for import and version
- ✅ Set up CI (GitHub Actions)
- ✅ Configure black, ruff, pytest

## Done
 - Scaffold package skeleton and tooling
 - Implemented CoreClient sync methods (embeddings, similarity, themes, sentiment)
 - Developed Job model with refresh/wait polling
 - Defined Pydantic API models and exception hierarchy
 - Set up CI (GitHub Actions), pre-commit (black, ruff), pytest configuration
 - Wrote tests for import/version and CoreClient behavior
 - Added Analyzer with execution flow, result wrapping, and caching support
 - Created built-in Process classes: ThemeGeneration, SentimentProcess, ThemeAllocation, Cluster
 - Implemented result helpers: ThemeGenerationResult, SentimentResult
 - Extended ThemeAllocationResult (assign_single, assign_multi, heatmap) with similarity support
 - Extended ClusterResult (kmeans, dbscan, plot_scatter, dendrogram)

## Next
1. Build CLI interface and entry points for command-line usage (defer to later)
2. Implement OAuth2Credentials and CoreAsyncClient for async workflows (defer to later)
3. Integrate on-disk caching in Analyzer and enable persistent memoization
4. Add concurrency/async support to Analyzer.run (parallel process execution)
5. Enhance allocation and clustering processes: custom thresholds, distance metrics, additional algorithms (low priority)
6. Develop user documentation, Jupyter notebooks, and examples

 ## Considerations
 - Packaging: pyproject.toml vs setup.py
 - Type compatibility for Python >=3.8
 - Pre-commit configuration
 - Pydantic v2 usage

 ## Revisit Later
 - Async client
 - OAuth2 credentials
 - Caching layer
 - Analyzer high-level API
 - CLI interface