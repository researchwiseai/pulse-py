# Changelog - OpenAPI v0.9.0 Update

## Version 0.9.0 - Enhanced API Features and Breaking Changes

**Release Date:** [TBD]

This major release introduces significant enhancements to the Pulse SDK, implementing OpenAPI specification v0.9.0 with new features, improved functionality, and some breaking changes. As the SDK is in beta, these changes are designed to improve the overall developer experience and API consistency.

---

## 🚀 New Features

### Enhanced Theme Generation
- **Interactive Mode**: New `interactive` parameter enables interactive theme generation
- **Theme Sets**: Version `2025-09-01` returns `ThemeSetsResponse` with multiple theme sets
- **Context Steering**: `context` parameter to guide theme generation with domain-specific context
- **Pruning**: `prune` parameter to remove lowest-frequency themes (0-25 threshold)
- **Initial Sets**: `initial_sets` parameter for multiple theme set generation (requires `interactive=True`)
- **Enhanced Theme Model**: Themes now include `shortLabel`, `label`, `description`, and exactly 2 `representatives`

### Advanced Clustering Algorithms
- **Spherical K-Means**: `algorithm="skmeans"` for normalized vector clustering
- **Agglomerative Clustering**: `algorithm="agglomerative"` for hierarchical clustering
- **HDBSCAN**: `algorithm="hdbscan"` for density-based clustering with noise detection
- **Algorithm Selection**: Default remains `"kmeans"` with explicit algorithm specification
- **Enhanced Response**: `ClusteringResponse` now includes `algorithm` field and structured `clusters`

### Text Splitting for Similarity Analysis
- **Multiple Units**: Support for `sentence`, `newline`, and `word` level splitting
- **Advanced Aggregation**: `mean`, `max`, `top2`, `top3` aggregation methods
- **Sliding Windows**: `window_size` and `stride_size` parameters for window processing
- **Flexible Configuration**: Per-set splitting configuration with `Split` and `UnitAgg` models
- **Enhanced Limits**: Improved input validation for sync (500) and async (44,721) modes

### Extraction Type Control
- **Type Selection**: `type` parameter with `"named-entities"` (default) or `"themes"`
- **Dictionary Expansion**: `expand_dictionary` with configurable `expand_dictionary_limit`
- **Validation Rules**: `expand_dictionary=False` required when `type="themes"`
- **Simplified API**: Streamlined parameters removing deprecated fields

### Usage Estimation
- **No Authentication**: New `/usage/estimate` endpoint works without authentication
- **Feature Support**: Estimate usage for all major features
- **Planning Tool**: Perfect for budgeting and capacity planning

---

## 🔧 Improvements

### Input Validation and Limits
- **Stricter Validation**: Enhanced input validation with clear error messages
- **Sync vs Async Limits**: Different limits for synchronous and asynchronous processing
- **Cross-field Validation**: Validation of parameter interdependencies
- **Better Error Context**: Field-level error details with paths and locations

### Enhanced Error Handling
- **Structured Errors**: New `ErrorResponse` and `ErrorDetail` models
- **Field-level Details**: Specific error information for validation failures
- **Error Paths**: Clear indication of which fields caused validation errors
- **Improved Messages**: More actionable error messages for developers

### Response Model Enhancements
- **Consistent Structure**: Improved consistency across all response models
- **Better Typing**: Enhanced type safety with Pydantic v2+ features
- **Usage Tracking**: Updated `UsageRecord` model with `quantity` field

---

## ⚠️ Breaking Changes

### Field Renames
- **UsageRecord**: `units` field renamed to `quantity`
  - **Migration**: Use `quantity` instead of `units`
  - **Compatibility**: `units` property maintained for backward compatibility (deprecated)

### Theme Model Changes
- **New Structure**: Themes now require `shortLabel`, `label`, `description`, `representatives`
- **Representatives**: Exactly 2 representative strings required (was variable)
- **Enhanced Metadata**: More detailed theme information for better UX

### Response Type Changes
- **ThemeSetsResponse**: New response type when `version="2025-09-01"`
- **Conditional Returns**: Theme endpoints may return different response types based on version
- **Type Handling**: Applications should handle both `ThemesResponse` and `ThemeSetsResponse`

### Extractions API Simplification
- **Parameter Changes**:
  - `texts` → `inputs`
  - `categories` → `dictionary` (flattened format)
  - Removed: `use_ner`, `use_llm`, `threshold`
- **New Parameters**: `type`, `expand_dictionary`, `expand_dictionary_limit`
- **Validation**: New cross-field validation rules

### Input Limits Updates
- **New Sync Limits**: Updated synchronous processing limits
- **Async Limits**: New asynchronous processing limits
- **Cross-product Limits**: Similarity cross-product limited to 20,000 for sync mode

---

## 📊 Input Validation Limits

### Synchronous Mode (fast=True)
| Endpoint | Limit | Notes |
|----------|-------|-------|
| Embeddings | 200 texts | - |
| Similarity (self) | 500 texts | - |
| Similarity (cross) | 20,000 cross-product | \|set_a\| × \|set_b\| ≤ 20,000 |
| Themes | 200 texts | - |
| Clustering | 500 texts | - |
| Sentiment | 200 texts | - |
| Extractions | 200 texts | - |

### Asynchronous Mode (fast=False)
| Endpoint | Limit | Notes |
|----------|-------|-------|
| Embeddings | 5,000 texts | - |
| Similarity | 44,721 texts | - |
| Themes | 500 texts | - |
| Clustering | 44,721 texts | - |
| Sentiment | 5,000 texts | - |
| Extractions | 5,000 texts | - |

---

## 🔄 Migration Guide

### Immediate Actions Required

1. **Update UsageRecord References**
   ```python
   # Before
   record.units

   # After
   record.quantity  # or record.units (deprecated)
   ```

2. **Handle New Theme Structure**
   ```python
   # Before
   theme.name
   theme.examples

   # After
   theme.shortLabel
   theme.label
   theme.description
   theme.representatives  # exactly 2 items
   ```

3. **Update Extractions Calls**
   ```python
   # Before
   client.extract_elements(
       texts=texts,
       categories=["cat1", "cat2"],
       use_ner=True
   )

   # After
   client.extract_elements(
       inputs=texts,
       dictionary=["term1", "term2", "term3"],
       type="named-entities",
       expand_dictionary=True
   )
   ```

### Optional Enhancements

1. **Leverage New Clustering Algorithms**
   ```python
   # Enhanced clustering
   response = client.cluster_texts(
       inputs=texts,
       k=5,
       algorithm="skmeans",  # or "agglomerative", "hdbscan"
       fast=True
   )
   ```

2. **Use Text Splitting for Similarity**
   ```python
   # Sentence-level similarity
   split_config = Split(set_a=UnitAgg(unit="sentence", agg="max"))
   response = client.compare_similarity(SimilarityRequest(
       set=texts,
       split=split_config,
       fast=True
   ))
   ```

3. **Enhanced Theme Generation**
   ```python
   # Interactive themes with context
   response = client.generate_themes(
       texts=texts,
       version="2025-09-01",
       interactive=True,
       initial_sets=2,
       context="Customer feedback analysis",
       prune=2,
       fast=True
   )
   ```

---

## 🧪 Testing Your Migration

### Validation Checklist
- [ ] Update all `units` references to `quantity`
- [ ] Handle new `Theme` model structure
- [ ] Update `extract_elements` parameter names
- [ ] Test with new input validation limits
- [ ] Handle potential `ThemeSetsResponse` returns
- [ ] Update error handling for new error structure

### Testing Strategy
1. **Start Small**: Test with minimal datasets first
2. **Validate Limits**: Ensure your inputs meet new validation requirements
3. **Error Handling**: Test error scenarios to verify new error structure handling
4. **Feature Testing**: Gradually adopt new features like clustering algorithms
5. **Performance Testing**: Test with maximum input sizes for your use cases

---

## 📚 Documentation Updates

### New Documentation
- [Migration Guide](docs/migration-v0.9.0.md) - Comprehensive migration instructions
- [Enhanced API Documentation](docs/core-client.md) - Updated with all new features
- [Model Documentation](docs/models.md) - Complete model reference
- [Jupyter Examples](examples/) - Updated notebooks showcasing new features

### Updated Examples
- `high_level_api.ipynb` - Enhanced with new clustering and theme features
- `low_level_api_v090.ipynb` - New notebook demonstrating v0.9.0 features
- `dsl_api.ipynb` - Updated with new algorithm options

---

## 🔧 Developer Experience Improvements

### Enhanced Error Messages
- Field-level validation errors with specific paths
- Clear indication of parameter interdependencies
- Actionable error messages for quick resolution

### Better Type Safety
- Enhanced Pydantic models with strict validation
- Improved IDE support with better type hints
- Runtime validation for all API parameters

### Improved Documentation
- Comprehensive examples for all new features
- Clear migration paths for breaking changes
- Enhanced API reference with usage examples

---

## 🚦 Version Compatibility

### Supported Versions
- **Python**: 3.8+ (unchanged)
- **Pydantic**: 2.0+ (unchanged)
- **API Version**: v0.9.0 (new)

### Backward Compatibility
- `units` property maintained on `UsageRecord` (deprecated)
- Legacy extraction parameters accepted but deprecated
- Existing theme processing continues to work with new structure

### Deprecation Timeline
- `UsageRecord.units`: Deprecated in v0.9.0, removal planned for v1.0.0
- Legacy extraction parameters: Deprecated in v0.9.0, removal planned for v1.0.0

---

## 🐛 Bug Fixes

- Fixed response parsing for large similarity matrices
- Improved job polling reliability for long-running tasks
- Enhanced error handling for network timeouts
- Fixed edge cases in theme allocation scoring

---

## 🎯 Performance Improvements

- Optimized request batching for large datasets
- Improved memory usage for similarity computations
- Enhanced caching for repeated operations
- Faster response parsing for structured data

---

## 🔮 Looking Forward

### Upcoming Features (v0.10.0)
- Additional clustering algorithms
- Enhanced text preprocessing options
- Improved batch processing capabilities
- Extended similarity metrics

### Long-term Roadmap
- Real-time streaming analysis
- Advanced visualization tools
- Multi-language support enhancements
- Enterprise security features

---

## 📞 Support and Feedback

### Getting Help
- **Documentation**: Updated comprehensive docs with examples
- **Migration Support**: Detailed migration guide with code examples
- **Error Resolution**: Enhanced error messages with clear resolution steps

### Reporting Issues
- Use enhanced error information when reporting bugs
- Include API version and feature context
- Leverage usage estimation for capacity planning

### Community
- Share feedback on new clustering algorithms
- Contribute examples for text splitting use cases
- Help improve documentation with real-world usage patterns

---

**Note**: This is a major release with breaking changes. Please review the migration guide carefully and test thoroughly before upgrading production systems. The enhanced features provide significant new capabilities while maintaining the core SDK philosophy of simplicity and power.
