# Error Recovery Guide

This comprehensive guide helps you understand, diagnose, and resolve errors when using the Pulse SDK. It covers error categories, specific error codes, recovery strategies, and best practices for robust error handling.

## Quick Reference

| Error Category | Severity | Typical Recovery | Auto-Retry |
|----------------|----------|------------------|------------|
| [Network Errors](#network-errors) | Transient | Retry with backoff | ✅ Yes |
| [Authentication Errors](#authentication-errors) | Permanent | Fix credentials | ❌ No |
| [API Rate Limiting](#api-rate-limiting) | Transient | Wait and retry | ✅ Yes |
| [Configuration Errors](#configuration-errors) | Permanent | Fix configuration | ❌ No |
| [API Errors](#api-errors) | Mixed | Depends on code | ⚠️ Selective |

## Error Categories and Codes

### Network Errors

Network errors are typically **transient** and should be retried with exponential backoff.

#### Connection Errors

**Error Code:** `NetworkError` / `ConnectError`
**HTTP Status:** N/A (Connection failed)
**Severity:** Transient

```python
import httpx
from pulse.core.exceptions import NetworkError

try:
    client = CoreClient.with_client_credentials()
    result = client.create_embeddings(["test"])
except NetworkError as e:
    print(f"Network error: {e}")
    # Automatic retry handled by SDK
```

**Common Causes:**
- DNS resolution failure
- Network connectivity issues
- Firewall blocking connections
- Server temporarily unavailable

**Recovery Strategies:**
1. **Automatic Retry** (SDK handles this)
   - 3 attempts with exponential backoff
   - Delays: 0.5s, 1s, 2s
2. **Manual Retry** with longer delays
3. **Check Network Configuration**
   - Verify internet connectivity
   - Check firewall settings
   - Validate DNS resolution

#### Timeout Errors

**Error Code:** `TimeoutError` / `ConnectTimeout` / `ReadTimeout` / `WriteTimeout`
**HTTP Status:** N/A (Request timed out)
**Severity:** Transient

```python
from pulse.core.exceptions import TimeoutError

try:
    # Large dataset that might timeout
    client = CoreClient.with_client_credentials()
    result = client.create_embeddings(large_text_list, fast=False)
except TimeoutError as e:
    print(f"Request timed out after {e.timeout}ms for {e.url}")
    # Consider breaking into smaller batches
```

**Recovery Strategies:**
1. **Increase Timeout**
   ```python
   import httpx

   # Custom timeout configuration
   timeout = httpx.Timeout(
       connect=30.0,    # 30s to establish connection
       read=300.0,      # 5 minutes to read response
       write=30.0,      # 30s to write request
       pool=10.0        # 10s to get connection from pool
   )

   client = CoreClient.with_client_credentials(timeout=timeout)
   ```

2. **Batch Processing**
   ```python
   # Break large requests into smaller chunks
   def process_in_batches(texts, batch_size=100):
       results = []
       for i in range(0, len(texts), batch_size):
           batch = texts[i:i + batch_size]
           try:
               result = client.create_embeddings(batch)
               results.extend(result.embeddings)
           except TimeoutError:
               # Retry with smaller batch
               smaller_batches = [batch[j:j+50] for j in range(0, len(batch), 50)]
               for small_batch in smaller_batches:
                   result = client.create_embeddings(small_batch)
                   results.extend(result.embeddings)
       return results
   ```

3. **Use Fast Mode**
   ```python
   # Use fast=True for quicker processing
   result = client.create_embeddings(texts, fast=True)
   ```

### Authentication Errors

Authentication errors are typically **permanent** and require fixing credentials or configuration.

#### Invalid Credentials

**Error Code:** `PulseAPIError` with status `401`
**HTTP Status:** `401 Unauthorized`
**Severity:** Permanent

```python
from pulse.core.exceptions import PulseAPIError

try:
    client = CoreClient.with_client_credentials()
    result = client.create_embeddings(["test"])
except PulseAPIError as e:
    if e.status == 401:
        print(f"Authentication failed: {e.message}")
        if e.aws_www_authenticate:
            print(f"AWS hint: {e.aws_www_authenticate}")
        # Fix credentials and retry
```

**Common Causes:**
- Invalid `PULSE_CLIENT_ID` or `PULSE_CLIENT_SECRET`
- Expired or revoked credentials
- Incorrect authentication flow
- Wrong environment configuration

**Recovery Strategies:**
1. **Verify Environment Variables**
   ```bash
   echo $PULSE_CLIENT_ID
   echo $PULSE_CLIENT_SECRET
   echo $PULSE_BASE_URL
   ```

2. **Check Credential Validity**
   ```python
   from pulse.debug import enable_debug

   enable_debug()  # Enable debug logging
   client = CoreClient.with_client_credentials()

   # Check token status
   token_info = client.debug_auth_status()
   print(f"Has token: {token_info.has_token}")
   print(f"Is valid: {token_info.is_valid}")
   print(f"Is expired: {token_info.is_expired}")
   ```

3. **Test Different Authentication Methods**
   ```python
   # Try explicit credentials
   from pulse.auth import ClientCredentialsAuth

   auth = ClientCredentialsAuth(
       client_id="your_client_id",
       client_secret="your_client_secret"
   )
   client = CoreClient(auth=auth)
   ```

#### Token Expiry

**Error Code:** `PulseAPIError` with status `401` (token expired)
**HTTP Status:** `401 Unauthorized`
**Severity:** Transient (auto-recoverable)

```python
# SDK automatically handles token refresh
try:
    client = CoreClient.with_client_credentials()
    # Long-running process - token may expire
    for batch in large_dataset:
        result = client.create_embeddings(batch)
        # Token refresh happens automatically if needed
except PulseAPIError as e:
    if e.status == 401 and "expired" in e.message.lower():
        print("Token expired - SDK should auto-refresh")
        # This should rarely happen due to automatic refresh
```

**Recovery Strategies:**
1. **Automatic Refresh** (SDK default behavior)
   - SDK refreshes tokens 60 seconds before expiry
   - No manual intervention required

2. **Manual Token Refresh**
   ```python
   # Force token refresh
   client._auth._refresh_token()
   ```

### API Rate Limiting

Rate limiting errors are **transient** and should be handled with appropriate backoff strategies.

#### Rate Limit Exceeded

**Error Code:** `PulseAPIError` with status `429`
**HTTP Status:** `429 Too Many Requests`
**Severity:** Transient

```python
import time
from pulse.core.exceptions import PulseAPIError

def make_request_with_rate_limiting(client, texts):
    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            return client.create_embeddings(texts)
        except PulseAPIError as e:
            if e.status == 429:
                # Extract retry-after header if available
                retry_after = e.headers.get('retry-after')
                if retry_after:
                    delay = float(retry_after)
                else:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)

                print(f"Rate limited. Waiting {delay}s before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)
            else:
                raise

    raise Exception("Max retries exceeded for rate limiting")
```

**Recovery Strategies:**
1. **Respect Retry-After Header**
   ```python
   if e.status == 429:
       retry_after = e.headers.get('retry-after', '60')
       time.sleep(float(retry_after))
   ```

2. **Implement Exponential Backoff**
   ```python
   def exponential_backoff(attempt, base_delay=1.0, max_delay=300.0):
       delay = min(base_delay * (2 ** attempt), max_delay)
       jitter = delay * 0.1 * random.random()  # Add jitter
       return delay + jitter
   ```

3. **Reduce Request Rate**
   ```python
   import time

   # Add delays between requests
   for batch in batches:
       result = client.create_embeddings(batch)
       time.sleep(0.1)  # 100ms delay between requests
   ```

### Configuration Errors

Configuration errors are **permanent** and require fixing the configuration.

#### Invalid Environment Configuration

**Error Code:** `ValueError` / `ConfigurationError`
**HTTP Status:** N/A
**Severity:** Permanent

```python
try:
    client = CoreClient.with_client_credentials()
except ValueError as e:
    if "Client Secret is required" in str(e):
        print("Missing PULSE_CLIENT_SECRET environment variable")
        print("Set it with: export PULSE_CLIENT_SECRET=your_secret")
```

**Common Configuration Issues:**

1. **Missing Environment Variables**
   ```bash
   # Required variables
   export PULSE_CLIENT_ID=your_client_id
   export PULSE_CLIENT_SECRET=your_client_secret

   # Optional variables
   export PULSE_BASE_URL=https://pulse.researchwiseai.com
   export PULSE_AUDIENCE=https://pulse.researchwiseai.com
   ```

2. **Wrong Environment Settings**
   ```python
   # Check current configuration
   from pulse.config import BASE_URL, AUDIENCE, CLIENT_ID

   print(f"Base URL: {BASE_URL}")
   print(f"Audience: {AUDIENCE}")
   print(f"Client ID: {CLIENT_ID}")
   ```

3. **Invalid URLs or Endpoints**
   ```python
   # Validate configuration
   import httpx

   try:
       response = httpx.get(BASE_URL + "/health")
       print(f"API endpoint reachable: {response.status_code}")
   except Exception as e:
       print(f"API endpoint unreachable: {e}")
   ```

### API Errors

API errors can be either transient or permanent depending on the specific error code.

#### Client Errors (4xx)

**Error Code:** `PulseAPIError` with status `4xx`
**HTTP Status:** `400-499`
**Severity:** Permanent (usually)

```python
from pulse.core.exceptions import PulseAPIError

try:
    # Invalid request data
    result = client.create_embeddings([])  # Empty list
except PulseAPIError as e:
    if 400 <= e.status < 500:
        print(f"Client error {e.status}: {e.message}")
        # Fix the request and retry
```

**Common 4xx Errors:**

| Status | Error | Cause | Recovery |
|--------|-------|-------|----------|
| 400 | Bad Request | Invalid request format | Fix request data |
| 401 | Unauthorized | Invalid credentials | Fix authentication |
| 403 | Forbidden | Insufficient permissions | Check account permissions |
| 404 | Not Found | Invalid endpoint | Check API documentation |
| 422 | Unprocessable Entity | Invalid data format | Validate input data |

#### Server Errors (5xx)

**Error Code:** `PulseAPIError` with status `5xx`
**HTTP Status:** `500-599`
**Severity:** Transient (usually)

```python
try:
    result = client.create_embeddings(texts)
except PulseAPIError as e:
    if 500 <= e.status < 600:
        print(f"Server error {e.status}: {e.message}")
        # Retry with backoff
        time.sleep(5)
        result = client.create_embeddings(texts)
```

**Common 5xx Errors:**

| Status | Error | Cause | Recovery |
|--------|-------|-------|----------|
| 500 | Internal Server Error | Server-side issue | Retry after delay |
| 502 | Bad Gateway | Proxy/gateway error | Retry after delay |
| 503 | Service Unavailable | Server overloaded | Retry with longer delay |
| 504 | Gateway Timeout | Upstream timeout | Retry with longer delay |

## Error Severity Classification

### Transient Errors (Retry Recommended)

These errors are temporary and likely to succeed on retry:

- **Network connectivity issues** (`NetworkError`, `ConnectError`)
- **Timeout errors** (`TimeoutError`, `ReadTimeout`, `WriteTimeout`)
- **Rate limiting** (`429 Too Many Requests`)
- **Server errors** (`500`, `502`, `503`, `504`)
- **Token expiry** (`401` with expired token)

### Permanent Errors (Fix Required)

These errors require fixing the underlying issue:

- **Authentication failures** (`401` with invalid credentials)
- **Authorization failures** (`403 Forbidden`)
- **Invalid requests** (`400 Bad Request`, `422 Unprocessable Entity`)
- **Configuration errors** (`ValueError`, missing environment variables)
- **Resource not found** (`404 Not Found`)

## Multi-Error Prioritization

When multiple errors occur, handle them in this priority order:

### Priority 1: Configuration Errors
Fix these first as they prevent any requests from working:

```python
def diagnose_configuration():
    issues = []

    # Check required environment variables
    import os
    required_vars = ['PULSE_CLIENT_ID', 'PULSE_CLIENT_SECRET']
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing {var} environment variable")

    # Check API connectivity
    try:
        import httpx
        response = httpx.get(BASE_URL + "/health", timeout=10)
        if response.status_code != 200:
            issues.append(f"API health check failed: {response.status_code}")
    except Exception as e:
        issues.append(f"Cannot reach API endpoint: {e}")

    return issues
```

### Priority 2: Authentication Errors
Fix authentication before attempting API calls:

```python
def diagnose_authentication():
    try:
        client = CoreClient.with_client_credentials()
        token_info = client.debug_auth_status()

        if not token_info.has_token:
            return "No authentication token available"
        elif not token_info.is_valid:
            return "Authentication token is invalid"
        elif token_info.is_expired:
            return "Authentication token has expired"
        else:
            return None  # Authentication OK
    except Exception as e:
        return f"Authentication error: {e}"
```

### Priority 3: Network and API Errors
Handle these during normal operation:

```python
def handle_api_request(client, texts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.create_embeddings(texts)
        except PulseAPIError as e:
            if e.status == 429:  # Rate limiting
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            elif 500 <= e.status < 600:  # Server errors
                time.sleep(5)  # Fixed delay for server errors
                continue
            else:
                raise  # Permanent error, don't retry
        except (NetworkError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise Exception("Max retries exceeded")
```

## Comprehensive Error Handling Example

Here's a complete example that handles all error types:

```python
import time
import logging
from typing import List, Any
from pulse.core.client import CoreClient
from pulse.core.exceptions import PulseAPIError, NetworkError, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustPulseClient:
    def __init__(self, max_retries=3, base_delay=1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize client with error handling."""
        try:
            self.client = CoreClient.with_client_credentials()
            logger.info("Client initialized successfully")
        except ValueError as e:
            if "Client Secret is required" in str(e):
                logger.error("Missing PULSE_CLIENT_SECRET environment variable")
                raise ConfigurationError("Set PULSE_CLIENT_SECRET environment variable")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            raise

    def create_embeddings_robust(self, texts: List[str], **kwargs) -> Any:
        """Create embeddings with comprehensive error handling."""

        # Validate input
        if not texts:
            raise ValueError("Input texts cannot be empty")

        if len(texts) > 1000:
            logger.warning(f"Large batch size ({len(texts)}), consider splitting")

        # Attempt request with retries
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}")
                result = self.client.create_embeddings(texts, **kwargs)
                logger.info(f"Successfully processed {len(texts)} texts")
                return result

            except PulseAPIError as e:
                last_exception = e

                if e.status == 401:
                    # Authentication error
                    logger.error(f"Authentication failed: {e.message}")
                    if "expired" in e.message.lower():
                        logger.info("Token expired, attempting refresh")
                        self.client._auth._refresh_token()
                        continue
                    else:
                        raise AuthenticationError(f"Invalid credentials: {e.message}")

                elif e.status == 429:
                    # Rate limiting
                    retry_after = e.headers.get('retry-after', str(self.base_delay * (2 ** attempt)))
                    delay = float(retry_after)
                    logger.warning(f"Rate limited, waiting {delay}s")
                    time.sleep(delay)
                    continue

                elif 500 <= e.status < 600:
                    # Server error - retry
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Server error {e.status}, retrying in {delay}s")
                    time.sleep(delay)
                    continue

                else:
                    # Client error - don't retry
                    logger.error(f"Client error {e.status}: {e.message}")
                    raise ClientError(f"Request failed: {e.message}")

            except (NetworkError, TimeoutError) as e:
                last_exception = e

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Network/timeout error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Network error after {self.max_retries} attempts: {e}")
                    raise NetworkError(f"Network error after {self.max_retries} attempts: {e}")

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed. Last error: {last_exception}")

# Custom exception classes for better error handling
class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass

class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass

class ClientError(Exception):
    """Raised for client-side errors (4xx)."""
    pass

# Usage example
def main():
    try:
        client = RobustPulseClient(max_retries=5, base_delay=1.0)

        texts = ["Hello world", "How are you?", "This is a test"]
        result = client.create_embeddings_robust(texts, fast=True)

        print(f"Successfully created {len(result.embeddings)} embeddings")

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Please check your environment variables and try again")

    except AuthenticationError as e:
        print(f"Authentication error: {e}")
        print("Please check your credentials and try again")

    except ClientError as e:
        print(f"Client error: {e}")
        print("Please check your request data and try again")

    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    main()
```

## Debugging Tools

Use the SDK's built-in debugging tools to diagnose issues:

```python
from pulse.debug import enable_debug, get_debug_stats

# Enable comprehensive debugging
enable_debug(
    log_requests=True,
    log_responses=True,
    log_timing=True,
    log_auth_status=True
)

# Make requests and check statistics
client = CoreClient.with_client_credentials()
result = client.create_embeddings(["test"])

# Review debug statistics
stats = get_debug_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Failed requests: {stats.failed_requests}")
print(f"Success rate: {stats.success_rate:.1f}%")
print(f"Average request time: {stats.average_request_time:.3f}s")
```

## Best Practices

### 1. Always Use Timeouts
```python
import httpx

# Configure appropriate timeouts
timeout = httpx.Timeout(
    connect=30.0,    # Connection timeout
    read=300.0,      # Read timeout (5 minutes for large requests)
    write=30.0,      # Write timeout
    pool=10.0        # Pool timeout
)

client = CoreClient.with_client_credentials(timeout=timeout)
```

### 2. Implement Circuit Breaker Pattern
```python
import time
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

### 3. Log Errors Appropriately
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    result = client.create_embeddings(texts)
except PulseAPIError as e:
    logger.error(
        "API error occurred",
        extra={
            'status_code': e.status,
            'error_code': e.code,
            'message': e.message,
            'request_id': e.aws_request_id,
            'url': 'create_embeddings'
        }
    )
except Exception as e:
    logger.exception("Unexpected error occurred")
```

### 4. Monitor Error Rates
```python
from collections import defaultdict
import time

class ErrorMonitor:
    def __init__(self, window_size=300):  # 5 minute window
        self.window_size = window_size
        self.errors = defaultdict(list)

    def record_error(self, error_type):
        now = time.time()
        self.errors[error_type].append(now)

        # Clean old errors
        cutoff = now - self.window_size
        self.errors[error_type] = [t for t in self.errors[error_type] if t > cutoff]

    def get_error_rate(self, error_type):
        return len(self.errors[error_type]) / (self.window_size / 60)  # errors per minute

    def should_alert(self, error_type, threshold=5):
        return self.get_error_rate(error_type) > threshold
```

This comprehensive error recovery guide provides you with the tools and knowledge to handle any error scenario when using the Pulse SDK. Remember to always check the error category first, then apply the appropriate recovery strategy based on whether the error is transient or permanent.
