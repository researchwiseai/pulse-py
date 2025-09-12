# Jobs and Errors

## Jobs (module: `pulse.core.jobs`)

The `Job` model represents an asynchronous operation submitted to the API. Most client methods can return a `Job` when `await_job_result=False` or when using explicit async mode.

Fields:
- `id` (alias `jobId`)
- `status` (alias `jobStatus`) – one of `pending`, `completed`, `error`, `failed`
- `message: str | None` – error message when failed
- `result_url: str | None` – URL to retrieve results on completion

Methods:
- `refresh(max_retries=10, retry_delay=10.0) -> Job` – Polls `/jobs?jobId=...` and returns an updated `Job`. Retries on 404/500 with backoff.
- `wait(timeout=180.0) -> Any` – Blocks until completion and returns the final JSON result (follows `result_url` when present). Raises `RuntimeError` on job failure, `TimeoutError` (module-defined) on timeouts.
- `result(timeout=180.0)` – Alias for `wait`.

Example:
```python
job = client.generate_summary(["..."], "question", await_job_result=False)
result = job.wait()
```

## Errors (module: `pulse.core.exceptions`)

### `PulseAPIError`
Raised on non‑success HTTP responses when calling the Core API.

Attributes:
- `status: int`
- `code: str | None`
- `message: str | None`
- `body: Any` – Parsed JSON or raw text

`str(error)` renders a readable message: `"Pulse API Error <status> <code>: <message>"`.

### `TimeoutError`
Distinct from `asyncio.TimeoutError`. Used by internals when a request or job exceeds the allowed time.

### `NetworkError`
Raised by internals for transport‑level issues.

