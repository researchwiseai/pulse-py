# Pulse SDK Automatic Batching Support Report

## Executive Summary

The Pulse SDK has **limited automatic batching support** that varies significantly across layers and features. Only **Sentiment Analysis** and **Similarity** have automatic batching implementations, while other features rely on API limits and manual chunking strategies.

## Detailed Analysis by Layer and Feature

### 1. Core Client Layer (`pulse.core.client.CoreClient`)

#### ✅ **Similarity** - Full Automatic Batching
- **Implementation**: Dedicated `batch_similarity()` method with intelligent chunking
- **Trigger**: Automatically called when `fast=False` and input exceeds limits:
  - Self-similarity: >500 items
  - Cross-similarity: >20,000 total items (set_a × set_b)
- **Algorithm**:
  - Splits large requests into chunks under 10k item limit
  - Uses numpy for matrix stitching to reconstruct full results
  - Handles both self-similarity and cross-similarity scenarios
- **Dependencies**: Requires numpy (`pip install pulse-sdk[analysis]`)

#### ✅ **Sentiment Analysis** - Basic Automatic Batching
- **Implementation**: Uses `chunk_texts()` utility for simple chunking
- **Trigger**: When input exceeds limits:
  - Fast mode: >200 texts
  - Slow mode: >10,000 texts
- **Algorithm**: Sequential processing of chunks with result aggregation
- **No dependencies**: Uses built-in chunking utilities

#### ❌ **Embeddings** - No Automatic Batching
- **Limit**: 2,000 items (MAX_EMBEDDINGS)
- **Behavior**: Raises validation error if exceeded
- **Manual workaround**: Users must chunk manually

#### ❌ **Theme Generation** - No Automatic Batching
- **Limit**: 500 items (MAX_THEMES)
- **Behavior**: Automatic sampling instead of batching:
  - Fast mode: samples 200 items randomly
  - Slow mode: samples 500 items randomly
- **Note**: This is sampling, not batching - data is lost

#### ❌ **Clustering** - No Automatic Batching
- **Limit**: 500 items (MAX_CLUSTERING)
- **Behavior**: Raises validation error if exceeded

#### ❌ **Element Extraction** - No Automatic Batching
- **Limit**: No explicit limit in code, but API may have limits
- **Behavior**: Sends full request to API

#### ❌ **Summarization** - No Automatic Batching
- **Limit**: 5,000 items (MAX_SUMMARIES)
- **Behavior**: Sends full request to API

### 2. Analysis Layer (`pulse.analysis.analyzer.Analyzer`)

The Analyzer layer **inherits** the batching behavior from the CoreClient but adds **no additional batching logic**.

#### ✅ **Sentiment Analysis** - Inherited Batching
- Uses CoreClient's automatic chunking
- Same limits and behavior as CoreClient

#### ✅ **Similarity** - Inherited Batching
- Uses CoreClient's batch_similarity when needed
- Same intelligent chunking as CoreClient

#### ❌ **Theme Generation** - Inherited Sampling (Not Batching)
- Uses CoreClient's sampling approach
- Additional sampling in ThemeGeneration process:
  - Fast mode: 200 items
  - Slow mode: 1,000 items

#### ❌ **Other Features** - No Additional Batching
- Theme Allocation, Clustering, Extraction: No batching support

### 3. DSL Layer (`pulse.dsl.Workflow`)

The DSL layer **delegates** all processing to underlying layers and adds **no batching logic**.

#### ✅ **Sentiment Analysis** - Delegated Batching
- Calls CoreClient through processes
- Inherits automatic chunking behavior
- **Special feature**: Supports nested input structures with flatten/reconstruct

#### ✅ **Similarity** - Delegated Batching
- Inherits CoreClient's batch_similarity functionality

#### ❌ **Other Features** - No Batching
- All other features delegate without additional batching

### 4. Starters Layer (`pulse.starters`)

The Starters layer provides **convenience functions** that delegate to CoreClient with **no additional batching**.

#### ✅ **Sentiment Analysis** (`sentiment_analysis()`) - Delegated Batching
- Calls CoreClient.analyze_sentiment()
- Inherits automatic chunking
- Auto-determines fast/slow mode based on input size (≤200 = fast)

#### ✅ **Similarity** (`compare_similarity()`) - Delegated Batching
- Calls CoreClient.compare_similarity()
- Inherits batch_similarity functionality
- Auto-determines fast/slow mode based on input size

#### ❌ **Other Features** - No Batching
- `generate_themes()`, `cluster_analysis()`, `extract_elements()`, `summarize()`: No batching

## Summary Table

| Feature | CoreClient | Analyzer | DSL | Starters | Implementation |
|---------|------------|----------|-----|----------|----------------|
| **Similarity** | ✅ Full | ✅ Inherited | ✅ Inherited | ✅ Inherited | `batch_similarity()` + numpy |
| **Sentiment** | ✅ Basic | ✅ Inherited | ✅ Inherited | ✅ Inherited | `chunk_texts()` utility |
| **Embeddings** | ❌ None | ❌ None | ❌ None | ❌ None | Manual chunking required |
| **Themes** | ❌ Sampling | ❌ Sampling | ❌ Sampling | ❌ Sampling | Random sampling, not batching |
| **Clustering** | ❌ None | ❌ None | ❌ None | ❌ None | Manual chunking required |
| **Extraction** | ❌ None | ❌ None | ❌ None | ❌ None | Manual chunking required |
| **Summarization** | ❌ None | ❌ None | ❌ None | ❌ None | Manual chunking required |

## Recommendations

### Immediate Improvements Needed

1. **Add automatic batching for Embeddings** - High priority given 2k limit
2. **Add automatic batching for Clustering** - Medium priority
3. **Add automatic batching for Element Extraction** - Medium priority
4. **Replace Theme Generation sampling with true batching** - Consider if preserving all data is important

### Implementation Patterns

The SDK uses two batching patterns:
1. **Simple chunking** (`chunk_texts()`) - for sequential processing
2. **Intelligent batching** (`pulse.core.batching`) - for complex matrix operations

New batching implementations should follow these established patterns.
