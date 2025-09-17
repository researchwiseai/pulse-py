# Product Overview

Pulse SDK is an idiomatic, type-safe Python client for the Researchwise AI Pulse REST API. It provides text analysis capabilities including:

- **Embeddings & Similarity**: Create text embeddings and compare similarity between texts
- **Theme Analysis**: Generate themes from text data and allocate texts to themes
- **Sentiment Analysis**: Analyze sentiment in text data
- **Clustering**: Group similar texts using various algorithms
- **Summarization**: Generate summaries from text collections
- **Element Extraction**: Extract specific elements/categories from texts

## Key Features

- **Multi-level APIs**: Low-level CoreClient for direct API calls, high-level Analyzer for workflows, and DSL builder for custom pipelines
- **Built-in Caching**: On-disk and in-memory caching via diskcache
- **Usage Reporting**: All responses include usage metrics
- **Data Science Integration**: First-class interop with pandas, NumPy, and scikit-learn
- **Authentication**: OAuth2 support with Client Credentials and Authorization Code PKCE flows
- **Async Job Support**: Handle long-running operations with job polling

## Target Users

Data scientists, researchers, and developers who need to analyze text data at scale using AI-powered analysis tools.