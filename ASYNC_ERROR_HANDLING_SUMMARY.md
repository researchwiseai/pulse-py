# Async Error Handling and Timeout Patterns Implementation

## Overview

This implementation adds comprehensive async error handling and timeout patterns to the Pulse SDK, fulfilling task 12 of the async-await-support specification. The enhancement provides proper asyncio.TimeoutError handling, asyncio.CancelledError support, async-specific error recovery patterns, timeout context managers, and proper exception chaining for async HTTP errors.

## Key Components Implemented

### 1. Enhanced Error Classes

#### AsyncTimeoutError
- Extends `asyncio.TimeoutError` with enhanced context information
- Includes operation name, timeout duration, and contextual data
- Provides detailed error messages for debugging

#### AsyncCancellationError
- Extends `asyncio.CancelledError` with enhanced context information
- Includes operation name and contextual data
- Helps identify where cancellation occurred in complex async workflows

### 2. Async Timeout Context Manager

#### async_timeout_context()
- Compatible with Python 3.10+ (uses asyncio.wait_for internally)
- Provides enhanced error reporting with operation names and context
- Supports both raising enhanced errors and warning-only modes
- Used throughout async components for consistent timeout handling

### 3. Async Error Recovery Patterns

#### AsyncErrorRecovery Class
- `with_timeout_and_cancellation()`: Execute coroutines with enhanced timeout/cancellation handling
- `with_retry_and_timeout()`: Retry logic with exponential backoff and timeout support
- `gather_with_error_handling()`: Enhanced asyncio.gather with timeout and error handling
- Configurable timeout behaviors and retry strategies

#### AsyncJobTimeoutManager Class
- Specialized timeout management for AsyncJob operations
- `wait_with_timeout()`: Enhanced job waiting with timeout context
- `wait_for_status_with_timeout()`: Status-specific waiting with timeout
- `cancel_with_timeout()`: Job cancellation with timeout handling

### 4. HTTP Error Handling Decorator

#### @handle_async_http_errors
- Converts httpx exceptions to appropriate Pulse SDK exceptions
- Handles httpx.TimeoutException → AsyncTimeoutError
- Handles httpx.ConnectError → NetworkError
- Handles httpx.HTTPStatusError → PulseAPIError
- Preserves asyncio.CancelledError without modification
- Provides proper exception chaining for debugging

### 5. Enhanced Async Components

#### AsyncJob Enhancements
- All methods now use enhanced timeout and cancellation handling
- `wait()`, `wait_with_callback()`, `wait_for_status()` use timeout contexts
- `refresh()` includes proper cancellation handling
- `cancel()` includes timeout handling for the cancellation request itself

#### AsyncCoreClient Enhancements
- `_request()` method includes enhanced HTTP error handling
- Proper cancellation handling for HTTP requests
- Enhanced error context for debugging

#### AsyncConcurrentConfig Enhancements
- `gather_jobs()` uses enhanced timeout and error handling
- Proper task cancellation on timeout or cancellation
- Enhanced error reporting for concurrent operations

#### Async Retry Enhancements
- `async_retry_request_enhanced()` includes cancellation handling
- Enhanced timeout error reporting
- Proper exception chaining for retry failures

### 6. Enhanced Async Components Integration

#### AsyncAnalyzer
- `run()` method includes comprehensive timeout handling
- Process-level error handling with context
- Proper cancellation propagation through analysis workflows

#### Async Starters
- `generate_themes_async()` enhanced with timeout and error handling
- Adaptive timeout calculation based on data size and mode
- Proper cancellation handling and context reporting

### 7. Comprehensive Test Suite

#### test_async_error_handling.py
- Tests for all new error classes and their functionality
- Timeout context manager tests
- Error recovery pattern tests
- HTTP error handling decorator tests
- Job timeout manager tests
- Convenience function tests
- Integration tests for real-world scenarios

## Key Features

### 1. Python 3.10+ Compatibility
- Uses `asyncio.wait_for` instead of `asyncio.timeout` (Python 3.11+)
- Maintains compatibility with existing Python 3.10 environments
- Proper async context manager implementation

### 2. Enhanced Error Context
- All timeout errors include operation names and contextual data
- Cancellation errors provide information about where cancellation occurred
- HTTP errors include proper exception chaining for debugging

### 3. Configurable Timeout Behavior
- `AsyncTimeoutConfig` class for centralized timeout configuration
- Different timeout values for different operation types
- Adaptive timeout calculation for data-dependent operations

### 4. Proper Exception Chaining
- All enhanced exceptions properly chain from original exceptions
- Maintains full stack traces for debugging
- Preserves original error information while adding context

### 5. Cancellation-Safe Operations
- All async operations properly handle `asyncio.CancelledError`
- Cancellation propagates correctly through nested operations
- Resource cleanup on cancellation

## Usage Examples

### Basic Timeout Handling
```python
from pulse.core.async_error_handling import with_async_timeout, AsyncTimeoutError

try:
    result = await with_async_timeout(
        some_long_operation(),
        timeout=30.0,
        operation="data_processing"
    )
except AsyncTimeoutError as e:
    print(f"Operation {e.operation} timed out after {e.timeout}s")
```

### Enhanced Job Waiting
```python
from pulse.core.async_jobs import AsyncJob
from pulse.core.async_error_handling import AsyncTimeoutError, AsyncCancellationError

try:
    result = await job.wait(timeout=60.0)
except AsyncTimeoutError as e:
    print(f"Job {e.context['job_id']} timed out")
except AsyncCancellationError as e:
    print(f"Job wait was cancelled: {e.context}")
```

### Concurrent Operations with Error Handling
```python
from pulse.core.async_error_handling import gather_with_timeout

results = await gather_with_timeout(
    operation1(),
    operation2(),
    operation3(),
    timeout=120.0,
    return_exceptions=True
)
```

## Requirements Fulfilled

### Requirement 5.1: Async Context Management
✅ Implemented async context managers with proper resource cleanup
✅ Enhanced timeout context managers for all async operations
✅ Proper exception handling in async contexts

### Requirement 5.2: Resource Cleanup
✅ Proper async resource cleanup on timeout and cancellation
✅ HTTP connection cleanup on errors
✅ Task cancellation on timeout

### Requirement 5.3: Exception Handling
✅ Proper asyncio.TimeoutError handling throughout async components
✅ asyncio.CancelledError support for job cancellation
✅ Async-specific error recovery patterns
✅ Timeout context managers for async operations
✅ Proper exception chaining for async HTTP errors

## Integration Points

The enhanced error handling is integrated throughout the async SDK:

1. **AsyncJob**: All job operations use enhanced timeout and cancellation handling
2. **AsyncCoreClient**: HTTP requests include enhanced error handling
3. **AsyncAnalyzer**: Analysis workflows include comprehensive error handling
4. **Async Starters**: All starter functions include timeout and error handling
5. **Async Concurrent**: Concurrent operations include enhanced error handling
6. **Async Retry**: Retry logic includes enhanced error handling

## Testing

The implementation includes comprehensive tests covering:
- Error class functionality and context handling
- Timeout context manager behavior
- Error recovery patterns and retry logic
- HTTP error handling and exception conversion
- Job timeout management
- Integration scenarios with real async operations

All tests pass and demonstrate the enhanced error handling capabilities work correctly with Python 3.10+.

## Backward Compatibility

The implementation is fully backward compatible:
- Existing async code continues to work unchanged
- Enhanced error handling is additive, not breaking
- Original exception types are still raised when appropriate
- New error types extend existing asyncio exceptions

This implementation provides a robust foundation for async error handling and timeout management throughout the Pulse SDK, enabling developers to build reliable async applications with proper error handling and resource management.
