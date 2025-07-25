# Repository Guidelines

This repository contains **pulse-sdk**, a Python client for the Researchwise AI Pulse REST API.

## Setup
- Use Python 3.8+ and create a virtual environment.
- Install dependencies with:
  ```bash
  pip install -e .[dev]
  ```
- Install pre-commit hooks:
  ```bash
  pre-commit install
  ```

## Running Tests
- Standard tests:
  ```bash
  make test
  ```
  or directly `pytest`.
- To re-record HTTP cassettes, run:
  ```bash
  make vcr-record
  ```
- CI runs pytest with:
  ```bash
  pytest -q --disable-warnings --maxfail=1 --vcr-record=none
  ```
- Many tests require OAuth credentials; set the following environment variables:
  - `PULSE_CLIENT_ID`
  - `PULSE_CLIENT_SECRET`
  - Optional: `PULSE_TOKEN_URL`, `PULSE_AUDIENCE`

## Formatting and Linting
- Format Python code with **black** (line length 88):
  ```bash
  black .
  ```
- Format notebooks with **nbqa black**:
  ```bash
  nbqa black .
  ```
- Run **ruff** for linting:
  ```bash
  ruff check pulse tests
  ```
- These commands are run automatically by pre-commit.

## Notes
- Keep changes backward compatible with existing models and APIs.
- Avoid committing large datasets or generated notebook outputs.
