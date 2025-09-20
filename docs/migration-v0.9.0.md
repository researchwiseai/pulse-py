# Migration Guide: OpenAPI v0.9.0 Updates

This guide helps you migrate from previous versions of the Pulse SDK to the new OpenAPI v0.9.0 specification. While the SDK is in beta and breaking changes are expected, this guide provides clear instructions for updating your code.

## Overview of Changes

The v0.9.0 update introduces several enhancements and breaking changes:

- **Enhanced Themes**: New interactive mode, theme sets, and pruning options
- **Advanced Clustering**: Multiple algorithms (spherical k-means, agglomerative, HDBSCAN)
- **Text Splitting**: Enhanced similarity analysis with sentence/word splitting
- **Extraction Types**: Type control for named entities vs themes
- **Usage Estimation**: New endpoint for credit estimation without authentication
- **Breaking Changes**: Field renames and model structure updates

## Breaking Changes

### 1. UsageRecord Field Rename

**Change**: The `units` field in `UsageRecord` has been renamed to `quantity`.

**Before**:
```python
from pulse.core.models import UsageRecord

record = UsageRecord(feature="embeddings", units=100)
print(record.units)  # 100
```

**After**:
```python
from pulse.core.models import UsageRecord

record = UsageRecord(feature="embeddings", quantity=100)
print(record.quantity)  # 100
print(record.units)     # 100 (backward compatibility property)
```

**Migration**: Update your code to use `quantity` instead of `units`. The `units` property is still available for backward compatibility but is deprecated.

### 2. Theme Model Structure Changes

**Change**: The `Theme` model now requires `shortLabel`, `label`, `description`, and exactly 2 `representatives`.

**Before**:
```python
# Old theme structure (hypothetical)
theme = {
    "name": "Food Quality",
    "summary": "Comments about food",
    "examples": ["great food", "tasty meal", "delicious"]
}
```

**After**:
```python
from pulse.core.models import Theme

theme = Theme(
    shortLabel="Food Quality",
    label="Food Quality and Taste",
    description="Customer feedback about food quality and taste.",
    representatives=["great food", "tasty meal"]  # Exactly 2 items required
)
```

**Migration**: Update any code that creates or processes `Theme` objects to use the new structure.

### 3. Themes Response Type Changes

**Change**: When using `version="2025-09-01"`, the themes endpoint returns `ThemeSetsResponse` instead of `ThemesResponse`.

**Before**:
```python
resp = client.generate_themes(texts, fast=True)
for theme in resp.themes:
    print(theme.label)
```

**After**:
```python
# Standard response (unchanged)
resp = client.generate_themes(texts, fast=True)
for theme in resp.themes:
    print(theme.shortLabel, theme.label)

# New theme sets response (version 2025-09-01)
resp = client.generate_themes(texts, version="2025-09-01", interactive=True, fast=True)
for i, theme_set in enumerate(resp.themeSets):
    print(f"Theme Set {i+1}:")
    for theme in theme_set:
        print(f"  {theme.shortLabel}: {theme.description}")
```

**Migration**:
- Update theme processing code to handle the new `Theme` structure
- For interactive themes, handle `ThemeSetsResponse` with multiple theme sets
- Use type checking to handle both response types if needed

### 4. Extractions API Changes

**Change**: The extractions API has been simplified and enhanced with type control.

**Before**:
```python
resp = client.extract_elements(
    texts=["The food was great"],
    categories=["food", "service"],
    dictionary={"food": ["food", "meal"], "service": ["service", "staff"]},
    use_ner=True,
    use_llm=False,
    threshold=0.5
)
```

**After**:
```python
resp = client.extract_elements(
    inputs=["The food was great"],
    dictionary=["food", "meal", "service", "staff"],
    type="named-entities",
    expand_dictionary=True,
    expand_dictionary_limit=10
)
```

**Migration**:
- Replace `texts` parameter with `inputs`
- Replace `categories` parameter with `dictionary` (flatten dictionary values)
- Remove `use_ner`, `use_llm`, and `threshold` parameters
- Use `type` parameter to control extraction behavior
- Use `expand_dictionary` instead of providing pre-expanded dictionaries

## New Features

### 1. Enhanced Themes Functionality

```python
# Interactive themes with multiple sets
resp = client.generate_themes(
    texts=["sample text 1", "sample text 2"],
    version="2025-09-01",
    interactive=True,
    initial_sets=2,
    min_themes=2,
    max_themes=10,
    context="Customer feedback analysis",
    prune=3,
    fast=True
)

# Access multiple theme sets
for i, theme_set in enumerate(resp.themeSets):
    print(f"Theme Set {i+1}: {len(theme_set)} themes")
```

### 2. Advanced Clustering Algorithms

```python
# Spherical k-means for normalized vectors
resp = client.cluster_texts(
    inputs=["text1", "text2", "text3"],
    k=2,
    algorithm="skmeans",
    fast=True
)

# HDBSCAN for density-based clustering
resp = client.cluster_texts(
    inputs=["text1", "text2", "text3"],
    k=2,
    algorithm="hdbscan",
    fast=True
)

print(f"Used algorithm: {resp.algorithm}")
```

### 3. Text Splitting for Similarity

```python
from pulse.core.models import SimilarityRequest, Split, UnitAgg

# Sentence-level splitting with max aggregation
split_config = Split(
    set_a=UnitAgg(unit="sentence", agg="max", window_size=2, stride_size=1)
)

resp = client.compare_similarity(SimilarityRequest(
    set_a=["First sentence. Second sentence."],
    set_b=["Another sentence. Final sentence."],
    split=split_config,
    fast=True
))

# Word-level splitting with top2 aggregation
word_split = Split(
    set_a=UnitAgg(unit="word", agg="top2")
)
```

### 4. Usage Estimation

```python
# Estimate credits without authentication
estimate = client.estimate_usage(
    feature="themes",
    inputs=["sample text 1", "sample text 2", "sample text 3"]
)
print(f"Estimated usage: {estimate.usage}")
```

## Input Validation Changes

The new version includes stricter input validation:

### Sync vs Async Limits

| Endpoint | Sync Limit | Async Limit |
|----------|------------|-------------|
| Embeddings | 200 | 5,000 |
| Similarity (self) | 500 | 44,721 |
| Similarity (cross) | 20,000 cross-product | No limit |
| Themes | 200 | 500 |
| Clustering | 500 | 44,721 |
| Sentiment | 200 | 5,000 |
| Extractions | 200 | 5,000 |

### Cross-field Validation

Some parameters now have interdependencies:

```python
# This will raise a validation error
themes_request = ThemesRequest(
    inputs=["text1", "text2"],
    interactive=False,
    initialSets=2  # Error: initialSets > 1 requires interactive=True
)

# This will raise a validation error
extractions_request = ExtractionsRequest(
    inputs=["text1"],
    dictionary=["term1", "term2", "term3"],
    type="themes",
    expand_dictionary=True  # Error: expand_dictionary must be False for type="themes"
)
```

## Error Handling Improvements

The new version provides more detailed error information:

```python
from pulse.core.exceptions import PulseAPIError

try:
    resp = client.generate_themes(["single text"])  # Too few inputs
except PulseAPIError as e:
    print(f"Error code: {e.code}")
    print(f"Error message: {e.message}")

    # New: detailed field-level errors
    if hasattr(e, 'errors') and e.errors:
        for error in e.errors:
            print(f"Field error: {error.field} - {error.message}")
            if error.path:
                print(f"Error path: {error.path}")
```

## Compatibility Notes

### Backward Compatibility

- The `units` property on `UsageRecord` is maintained for backward compatibility
- Legacy field names in `ExtractionsRequest` are still accepted but deprecated
- Existing theme processing code will work but should be updated for new fields

### Version Pinning

To maintain compatibility with existing code while testing new features:

```python
# Use specific version for new features
resp = client.generate_themes(
    texts=["text1", "text2"],
    version="2025-09-01",  # Explicit version for new response format
    interactive=True
)

# Default behavior remains unchanged for existing code
resp = client.generate_themes(["text1", "text2"])  # Returns ThemesResponse
```

## Testing Your Migration

1. **Update Dependencies**: Ensure you're using the latest SDK version
2. **Run Existing Tests**: Verify that existing functionality still works
3. **Update Field References**: Replace `units` with `quantity` in usage tracking
4. **Test New Features**: Gradually adopt new parameters and response types
5. **Handle Response Types**: Update theme processing to handle new structure
6. **Validate Inputs**: Ensure your inputs meet the new validation requirements

## Getting Help

If you encounter issues during migration:

1. Check the updated API documentation
2. Review the error messages for specific field-level guidance
3. Use the usage estimation endpoint to validate your inputs
4. Test with small datasets first before processing large volumes

The enhanced error handling in v0.9.0 provides much more detailed feedback to help identify and resolve issues quickly.
