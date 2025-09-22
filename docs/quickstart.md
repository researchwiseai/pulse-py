# Quick Start Guide

Get up and running with the Pulse SDK in under 5 minutes! This guide will take you from installation to your first API call with minimal setup.

## 🚀 5-Minute Setup

### Step 1: Install the SDK (30 seconds)

Choose your installation method based on your needs:

**Recommended Installation (All features)**
```bash
pip install pulse-sdk[all]
```

**Minimal Installation (API access only)**
```bash
pip install pulse-sdk[minimal]
```

**Custom Installation (Choose your features)**
```bash
# Data science workflow
pip install pulse-sdk[analysis,visualization,caching]

# Basic API with progress tracking
pip install pulse-sdk[minimal,progress]
```

**Development Installation (Contributors)**
```bash
git clone https://github.com/researchwiseai/pulse-py.git
cd pulse-py
pip install -e ".[dev]"
```

> 💡 **Need help choosing?** See our [complete installation guide](installation.md) for detailed explanations of each option.

### Step 2: Test Your Installation (30 seconds)

Verify everything works with a simple test:

```python
from pulse.core.client import CoreClient

# Quick test
client = CoreClient()
print("✅ Pulse SDK installed successfully!")
```

### Step 3: Your First API Call (2 minutes)

Let's analyze some text sentiment:

```python
from pulse.starters import sentiment_analysis

# Analyze sentiment of some sample texts
texts = [
    "I love this product! It's amazing.",
    "The service was terrible and slow.",
    "It's okay, nothing special."
]

# Make your first API call
results = sentiment_analysis(texts)

# See the results
for i, result in enumerate(results.results):
    print(f"Text: {texts[i]}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2f}")
    print("---")
```

### Step 4: Try More Features (2 minutes)

**Summarize Text**
```python
from pulse.starters import summarize

# Summarize customer feedback
feedback = [
    "Great food but slow service",
    "Amazing pizza, friendly staff",
    "Food was cold, waited too long"
]

summary = summarize(feedback, question="What do customers say about the restaurant?")
print("Summary:", summary.summary)
```

**Cluster Similar Texts**
```python
from pulse.starters import cluster_analysis

# Group similar comments
comments = [
    "Love the pizza here",
    "Pizza is delicious",
    "Service is too slow",
    "Waited forever for food",
    "Great atmosphere",
    "Nice ambiance"
]

clusters = cluster_analysis(comments, k=3)
print("Clusters:", clusters.clusters)
```

🎉 **Congratulations!** You've successfully made your first API calls with the Pulse SDK.

---

## 📚 Choose Your Learning Path

### 👶 Beginner Path: Start Simple

If you're new to text analysis or the Pulse SDK, follow this path:

1. **[Starters Guide](starters.md)** - Simple one-line functions for common tasks
2. **[Authentication](authentication.md)** - Set up secure API access
3. **[Core Client](core-client.md)** - Direct API calls with more control

**Beginner Example: Analyze Customer Reviews**
```python
from pulse.starters import sentiment_analysis, theme_allocation

# Load reviews from a file (CSV, TXT, Excel supported)
reviews = [
    "Food was great, service was slow",
    "Amazing pizza, will come back",
    "Too expensive for the quality"
]

# Analyze sentiment
sentiment_results = sentiment_analysis(reviews)

# Find themes
themes = theme_allocation(reviews, themes=["Food", "Service", "Price"])

# Print results
for i, review in enumerate(reviews):
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment_results.results[i].sentiment}")
    print(f"Themes: {themes.assign_single()[i]}")
    print("---")
```

### 🚀 Advanced Path: Build Workflows

If you're comfortable with Python and want to build complex analysis pipelines:

1. **[Analyzer](analyzer.md)** - Multi-step workflows with caching
2. **[DSL Builder](dsl.md)** - Declarative pipeline construction
3. **[Models](models.md)** - Understanding response structures

**Advanced Example: Complete Analysis Pipeline**
```python
from pulse.analysis.analyzer import Analyzer
from pulse.analysis.processes import ThemeGeneration, SentimentProcess

# Your dataset
reviews = [
    "The food was excellent but service was slow",
    "Great atmosphere, terrible food",
    "Perfect experience, will recommend"
]

# Build analysis pipeline
processes = [
    ThemeGeneration(min_themes=3, max_themes=5),
    SentimentProcess()
]

# Run analysis with caching
with Analyzer(dataset=reviews, processes=processes, cache_dir=".pulse_cache") as analyzer:
    results = analyzer.run()

# Access results
print("Generated Themes:")
for theme in results.theme_generation.themes:
    print(f"- {theme}")

print("\nSentiment Analysis:")
df = results.sentiment.to_dataframe()
print(df[['text', 'sentiment', 'confidence']])
```

---

## 🔧 Installation Options

The Pulse SDK offers flexible installation options to match your specific needs:

### Core Dependencies (Always Included)

- **httpx** - HTTP client for API requests
- **pydantic** - Data validation and serialization
- **typing-extensions** - Type hints for older Python versions

### Optional Feature Sets

**Analysis Features** (`pulse-sdk[analysis]`)
- **numpy** - Numerical computing for embeddings
- **pandas** - Data manipulation and DataFrame support
- **scikit-learn** - Machine learning utilities for clustering

**Visualization Features** (`pulse-sdk[visualization]`)
- **matplotlib** - Basic plotting and charts
- **seaborn** - Statistical visualizations

**Additional Features**
- **NLP** (`pulse-sdk[nlp]`) - TextBlob for text processing
- **Caching** (`pulse-sdk[caching]`) - Diskcache for performance
- **Progress** (`pulse-sdk[progress]`) - Progress bars for long operations

### Installation Commands by Use Case

```bash
# Complete experience (recommended)
pip install pulse-sdk[all]

# Minimal API access only
pip install pulse-sdk[minimal]

# Data science workflow
pip install pulse-sdk[analysis,visualization,caching]

# Web service integration
pip install pulse-sdk[minimal,progress]

# Development and testing
pip install pulse-sdk[dev]
```

> 📖 **Detailed Guide:** See our [complete installation guide](installation.md) for comprehensive options, troubleshooting, and version compatibility information.

---

## 🔐 Authentication Setup

### For Development/Testing (No Auth Required)

The SDK works without authentication in development mode:

```python
from pulse.core.client import CoreClient

# Uses default configuration
client = CoreClient()
```

### For Production Use

Set up authentication with your API credentials:

**Option 1: Environment Variables (Recommended)**
```bash
export PULSE_CLIENT_ID="your_client_id"
export PULSE_CLIENT_SECRET="your_client_secret"
```

```python
from pulse.core.client import CoreClient

# Automatically uses environment variables
client = CoreClient()
```

**Option 2: Direct Configuration**
```python
from pulse.core.client import CoreClient
from pulse.auth import ClientCredentialsAuth

auth = ClientCredentialsAuth(
    client_id="your_client_id",
    client_secret="your_client_secret"
)
client = CoreClient(auth=auth)
```

**Option 3: Interactive Authentication (PKCE)**
```python
from pulse.auth import AuthorizationCodePKCEAuth
from pulse.core.client import CoreClient

# Opens browser for interactive login
auth = AuthorizationCodePKCEAuth(client_id="your_client_id")
client = CoreClient(auth=auth)
```

---

## 📁 Working with Files

The SDK can process various file formats directly:

**Supported File Types:**
- `.txt` - Plain text files
- `.csv` - Comma-separated values
- `.tsv` - Tab-separated values
- `.xls`, `.xlsx` - Excel files

**Example: Analyze Reviews from CSV**
```python
from pulse.starters import sentiment_analysis

# CSV file with reviews in first column
results = sentiment_analysis("customer_reviews.csv")

# Process results
for result in results.results:
    print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
```

**Example: Load Custom Data**
```python
from pulse.starters import get_strings

# Load text from various sources
texts_from_file = get_strings("reviews.txt")
texts_from_list = get_strings(["review 1", "review 2", "review 3"])

print(f"Loaded {len(texts_from_file)} texts from file")
```

---

## 🎯 Common Use Cases

### Customer Feedback Analysis

```python
from pulse.starters import sentiment_analysis, theme_allocation

# Analyze customer feedback
feedback = [
    "Great product, fast shipping",
    "Product broke after one week",
    "Excellent customer service",
    "Too expensive for the quality"
]

# Get sentiment
sentiment = sentiment_analysis(feedback)

# Find themes
themes = theme_allocation(feedback, themes=["Product", "Shipping", "Service", "Price"])

# Combine results
for i, text in enumerate(feedback):
    print(f"Feedback: {text}")
    print(f"Sentiment: {sentiment.results[i].sentiment}")
    print(f"Main theme: {themes.assign_single()[i]}")
    print("---")
```

### Content Categorization

```python
from pulse.starters import cluster_analysis

# Categorize support tickets
tickets = [
    "Can't log into my account",
    "Forgot my password",
    "Product not working properly",
    "Defective item received",
    "How do I cancel my subscription?",
    "Want to upgrade my plan"
]

# Group similar tickets
clusters = cluster_analysis(tickets, k=3)

# Print grouped tickets
for cluster_id, texts in enumerate(clusters.clusters):
    print(f"Category {cluster_id + 1}:")
    for text in texts:
        print(f"  - {text}")
    print()
```

### Document Summarization

```python
from pulse.starters import summarize

# Summarize meeting notes
meeting_notes = [
    "Discussed Q4 budget allocation for marketing",
    "Decided to hire two new developers",
    "Marketing campaign launch delayed to January",
    "Need to review vendor contracts by month end"
]

# Generate summary
summary = summarize(
    meeting_notes,
    question="What were the key decisions and action items?",
    length="short"
)

print("Meeting Summary:")
print(summary.summary)
```

---

## 🔍 Next Steps

### Explore More Features

- **[Theme Analysis](analyzer.md#theme-generation)** - Discover topics in your text
- **[Similarity Comparison](core-client.md#similarity)** - Find similar documents
- **[Text Embeddings](core-client.md#embeddings)** - Convert text to vectors
- **[Element Extraction](core-client.md#extractions)** - Extract specific information

### Advanced Topics

- **[Caching](analyzer.md#caching)** - Speed up repeated analyses
- **[Batch Processing](core-client.md#batching)** - Handle large datasets efficiently
- **[Error Handling](jobs-and-errors.md)** - Robust error management
- **[Custom Workflows](dsl.md)** - Build complex analysis pipelines

### Get Help

- **[Troubleshooting](#troubleshooting)** - Common issues and solutions
- **[API Reference](core-client.md)** - Complete API documentation
- **[GitHub Issues](https://github.com/researchwiseai/pulse-py/issues)** - Report bugs or request features
- **[Examples](https://github.com/researchwiseai/pulse-py/tree/main/examples)** - Jupyter notebooks with detailed examples

---

## 🛠️ Troubleshooting

### Common Setup Issues

**Problem: Import Error**
```
ImportError: No module named 'pulse'
```
**Solution:**
```bash
# Make sure you're in the right environment
pip install pulse-sdk

# Or if using virtual environment:
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
pip install pulse-sdk
```

**Problem: Authentication Error**
```
PulseAPIError: 401 Unauthorized
```
**Solutions:**
1. **Check your credentials:**
   ```python
   import os
   print("Client ID:", os.getenv('PULSE_CLIENT_ID'))
   print("Has secret:", bool(os.getenv('PULSE_CLIENT_SECRET')))
   ```

2. **Verify environment:**
   ```python
   from pulse.config import get_config
   config = get_config()
   print("Environment:", config.env)
   print("Base URL:", config.base_url)
   ```

3. **Test without auth (development only):**
   ```python
   from pulse.core.client import CoreClient
   client = CoreClient()  # Uses default configuration
   ```

**Problem: Network/Timeout Errors**
```
httpx.ConnectTimeout: timed out
```
**Solutions:**
1. **Check internet connection**
2. **Increase timeout:**
   ```python
   from pulse.core.client import CoreClient
   import httpx

   # Custom timeout configuration
   timeout = httpx.Timeout(30.0)  # 30 seconds
   client = CoreClient(timeout=timeout)
   ```

3. **Use async jobs for large requests:**
   ```python
   # For large datasets, use async processing
   job = client.analyze_sentiment(large_text_list, await_job_result=False)
   result = job.wait()  # Poll until complete
   ```

**Problem: Dependency Conflicts**
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```
**Solutions:**
1. **Use fresh virtual environment:**
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install pulse-sdk
   ```

2. **Check for conflicts:**
   ```bash
   pip check
   ```

3. **Update pip:**
   ```bash
   pip install --upgrade pip
   ```

### Performance Issues

**Problem: Slow API Calls**
**Solutions:**
1. **Use fast mode for small datasets:**
   ```python
   # Automatically enabled for < 200 texts
   result = sentiment_analysis(texts)  # fast mode auto-detected by starters

   # Force fast mode
   result = client.analyze_sentiment(texts, fast=True)
   ```

2. **Enable caching:**
   ```python
   from pulse.analysis.analyzer import Analyzer

   with Analyzer(cache_dir=".pulse_cache") as analyzer:
       # Results are cached automatically
       results = analyzer.run()
   ```

3. **Use batch processing:**
   ```python
   # Process in smaller chunks
   import numpy as np

   texts = ["text"] * 1000
   chunks = np.array_split(texts, 10)  # 10 chunks of ~100 texts each

   all_results = []
   for chunk in chunks:
       results = sentiment_analysis(chunk.tolist())
       all_results.extend(results.results)
   ```

### Data Format Issues

**Problem: File Loading Errors**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solutions:**
1. **Check file path:**
   ```python
   import os
   print("Current directory:", os.getcwd())
   print("File exists:", os.path.exists("your_file.csv"))
   ```

2. **Use absolute paths:**
   ```python
   import os
   file_path = os.path.abspath("data/reviews.csv")
   results = sentiment_analysis(file_path)
   ```

3. **Verify file format:**
   ```python
   import pandas as pd

   # Check CSV structure
   df = pd.read_csv("reviews.csv")
   print("Columns:", df.columns.tolist())
   print("First few rows:")
   print(df.head())
   ```

### Getting More Help

If you're still having issues:

1. **Check the logs:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Your code here - will show detailed logs
   ```

2. **Create a minimal reproduction:**
   ```python
   from pulse.core.client import CoreClient

   client = CoreClient()
   try:
       result = client.analyze_sentiment(["test"])
       print("✅ Working!")
   except Exception as e:
       print(f"❌ Error: {e}")
   ```

3. **Report the issue:**
   - [GitHub Issues](https://github.com/researchwiseai/pulse-py/issues)
   - Include your Python version, OS, and error message
   - Provide a minimal code example that reproduces the issue

---

Ready to dive deeper? Check out our [complete documentation](index.md) or try the [interactive examples](https://github.com/researchwiseai/pulse-py/tree/main/examples)!
