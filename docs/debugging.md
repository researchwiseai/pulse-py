# Debugging and Introspection

The Pulse SDK includes comprehensive debugging and introspection tools to help you troubleshoot issues, monitor performance, and understand the SDK's behavior. These tools provide detailed insights into HTTP requests, authentication status, cache performance, and more.

## Quick Start

Enable debug mode by setting the `PULSE_DEBUG` environment variable:

```bash
export PULSE_DEBUG=true
python your_script.py
```

Or enable it programmatically:

```python
from pulse.debug import enable_debug

enable_debug()
```

## Debug Configuration

### Basic Configuration

```python
from pulse.debug import enable_debug, disable_debug, get_debug_config

# Enable with default settings
enable_debug()

# Check current configuration
config = get_debug_config()
print(f"Debug enabled: {config.enabled}")
print(f"Mask credentials: {config.mask_credentials}")

# Disable debug mode
disable_debug()
```

### Advanced Configuration

```python
enable_debug(
    log_requests=True,          # Log HTTP requests
    log_responses=True,         # Log HTTP responses
    log_timing=True,            # Log request timing
    log_cache_stats=True,       # Log cache hit/miss events
    log_auth_status=True,       # Log authentication events
    mask_credentials=True,      # Mask sensitive data in logs
    max_body_size=10000,        # Max characters for request/response bodies
    log_level="DEBUG",          # Logging level
    categories=["all"]          # Debug categories to enable
)
```

## Features

### 1. HTTP Request/Response Logging

When enabled, the debug system logs detailed information about all HTTP requests and responses:

```python
from pulse.core.client import CoreClient
from pulse.debug import enable_debug

enable_debug()

client = CoreClient.with_client_credentials()
# This will log the request and response details
response = client.create_embeddings(...)
```

**Example output:**
```
DEBUG [2024-01-15 10:30:15] pulse.debug - → POST https://pulse.researchwiseai.com/v1/embeddings
DEBUG [2024-01-15 10:30:15] pulse.debug -   Headers: {'Authorization': 'Bearer ***MASKED***', 'Content-Type': 'application/json'}
DEBUG [2024-01-15 10:30:15] pulse.debug -   Body: {"inputs": ["Hello world"], "fast": true}
DEBUG [2024-01-15 10:30:16] pulse.debug - ← 200 OK
DEBUG [2024-01-15 10:30:16] pulse.debug -   Body: {"results": [...], "usage": {...}}
DEBUG [2024-01-15 10:30:16] pulse.debug - ⏱️  POST https://pulse.researchwiseai.com/v1/embeddings completed in 0.847s
```

### 2. Authentication Token Inspection

Monitor authentication token status and lifecycle:

```python
from pulse.debug import inspect_token

client = CoreClient.with_client_credentials()
token_info = client.debug_auth_status()

print(f"Has token: {token_info.has_token}")
print(f"Token type: {token_info.token_type}")
print(f"Is valid: {token_info.is_valid}")
print(f"Is expired: {token_info.is_expired}")
print(f"Expires in: {token_info.expires_in} seconds")
print(f"Masked token: {token_info.masked_token}")
```

**Token refresh events are automatically logged:**
```
DEBUG [2024-01-15 10:30:20] pulse.debug - 🔄 Authentication token refreshed
```

### 3. Performance Timing

Track request performance and identify bottlenecks:

```python
from pulse.debug import get_debug_stats

# After making some requests
stats = get_debug_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Average request time: {stats.average_request_time:.3f}s")
print(f"Success rate: {stats.success_rate:.1f}%")

# View individual request timings
for timing in stats.request_timings[-5:]:  # Last 5 requests
    print(f"{timing.method} {timing.url} - {timing.duration:.3f}s")
```

### 4. Cache Hit/Miss Statistics

Monitor cache performance when using the Analyzer:

```python
from pulse.analysis.analyzer import Analyzer
from pulse.debug import enable_debug, get_debug_stats

enable_debug()

with Analyzer(dataset=texts, cache_dir=".pulse_cache") as analyzer:
    results = analyzer.run()  # First run - cache misses
    results = analyzer.run()  # Second run - cache hits

stats = get_debug_stats()
cache_stats = stats.cache_stats
print(f"Cache hit rate: {cache_stats.hit_rate:.1f}%")
print(f"Total cache requests: {cache_stats.total_requests}")
print(f"Cache hits: {cache_stats.hits}")
print(f"Cache misses: {cache_stats.misses}")
```

**Cache events are logged:**
```
DEBUG [2024-01-15 10:30:25] pulse.debug - 💾 Cache MISS: sentiment_process_abc123
DEBUG [2024-01-15 10:30:30] pulse.debug - 💾 Cache HIT: sentiment_process_abc123
```

### 5. Credential Masking

Sensitive information is automatically masked in logs:

```python
from pulse.debug import mask_credentials

# Automatically masks sensitive fields
sensitive_data = {
    "client_secret": "my_secret_key",
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "safe_field": "this_is_safe"
}

masked = mask_credentials(sensitive_data)
# Result: {"client_secret": "***MASKED***", "access_token": "***MASKED***", "safe_field": "this_is_safe"}
```

**Masked patterns include:**
- `client_secret`, `client_id`
- `access_token`, `refresh_token`
- `code`, `code_verifier` (PKCE)
- Bearer tokens in Authorization headers
- URL-encoded credentials

### 6. Granular Logging Categories

Control which types of debug information are logged:

```python
# Enable only specific categories
enable_debug(categories=["requests", "timing"])

# Available categories:
# - "all": Enable all logging (default)
# - "requests": HTTP request logging
# - "responses": HTTP response logging
# - "timing": Request timing information
# - "cache": Cache hit/miss events
# - "auth": Authentication events
# - "errors": Error logging
```

### 7. Error Tracking

Failed requests and errors are automatically tracked:

```python
stats = get_debug_stats()
print(f"Failed requests: {stats.failed_requests}")
print(f"Success rate: {stats.success_rate:.1f}%")
```

## Environment Variable Activation

The debug system can be automatically enabled via environment variables:

```bash
# Enable debug mode
export PULSE_DEBUG=true

# Alternative values that enable debug mode
export PULSE_DEBUG=1
export PULSE_DEBUG=yes
export PULSE_DEBUG=on
```

## Statistics Management

```python
from pulse.debug import get_debug_stats, clear_debug_stats

# Get current statistics
stats = get_debug_stats()

# Clear all statistics
clear_debug_stats()
```

## Integration with Core Components

The debug system is automatically integrated with:

- **CoreClient**: All HTTP requests are monitored
- **Authentication**: Token refresh events are logged
- **Analyzer**: Cache operations are tracked
- **Retry Logic**: Failed requests are counted

## Best Practices

### Development

```python
# Enable comprehensive debugging during development
enable_debug(
    log_requests=True,
    log_responses=True,
    log_timing=True,
    log_cache_stats=True,
    categories=["all"]
)
```

### Production Monitoring

```python
# Enable minimal debugging for production monitoring
enable_debug(
    log_requests=False,
    log_responses=False,
    log_timing=True,
    log_auth_status=True,
    categories=["timing", "auth", "errors"]
)
```

### Testing

```python
# Enable specific debugging for tests
enable_debug(
    log_requests=True,
    log_responses=False,  # Avoid cluttering test output
    mask_credentials=True,
    categories=["requests", "errors"]
)
```

## Performance Impact

The debug system is designed to have minimal performance impact:

- **Disabled by default**: No overhead when not enabled
- **Lazy evaluation**: Debug information is only computed when needed
- **Configurable verbosity**: Control the amount of logged information
- **Efficient masking**: Credential masking uses optimized regex patterns

## Troubleshooting Common Issues

### Authentication Problems

```python
# Check token status
client = CoreClient.with_client_credentials()
token_info = client.debug_auth_status()

if not token_info.is_valid:
    if not token_info.has_token:
        print("No authentication token available")
    elif token_info.is_expired:
        print("Authentication token has expired")
```

### Performance Issues

```python
# Analyze request timing
stats = get_debug_stats()
slow_requests = [t for t in stats.request_timings if t.duration > 5.0]

for timing in slow_requests:
    print(f"Slow request: {timing.method} {timing.url} - {timing.duration:.3f}s")
```

### Cache Issues

```python
# Check cache performance
stats = get_debug_stats()
if stats.cache_stats.hit_rate < 50:
    print("Low cache hit rate - consider cache configuration")
```

## Example: Complete Debug Session

```python
from pulse.debug import enable_debug, get_debug_stats, clear_debug_stats
from pulse.core.client import CoreClient
from pulse.analysis.analyzer import Analyzer

# Enable comprehensive debugging
enable_debug(
    log_requests=True,
    log_responses=True,
    log_timing=True,
    log_cache_stats=True,
    log_auth_status=True
)

# Clear previous statistics
clear_debug_stats()

try:
    # Create client and check auth status
    client = CoreClient.with_client_credentials()
    token_info = client.debug_auth_status()
    print(f"Authentication: {token_info.is_valid}")

    # Run analysis with caching
    texts = ["Sample text for analysis"]
    with Analyzer(dataset=texts, cache_dir=".pulse_cache") as analyzer:
        results = analyzer.run()

    # Review statistics
    stats = get_debug_stats()
    print(f"Total requests: {stats.total_requests}")
    print(f"Average time: {stats.average_request_time:.3f}s")
    print(f"Cache hit rate: {stats.cache_stats.hit_rate:.1f}%")

finally:
    client.close()
```

This comprehensive debugging system helps you understand exactly what's happening inside the Pulse SDK, making it easier to optimize performance, troubleshoot issues, and monitor your application's behavior.
