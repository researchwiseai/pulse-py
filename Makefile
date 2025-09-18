## Makefile for running tests and managing VCR cassettes

.PHONY: clean build test fmt lint vcr-clean vcr-record

# Clean build artifacts and cache files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.json

# Build distribution packages
build:
	python -m build

# Run all tests normally (uses record_mode='once' for VCR)
test:
	pytest

# Format code using black and nbqa
fmt:
	black .
	nbqa black .

# Lint code using ruff
lint:
	ruff check pulse tests

# Remove all recorded VCR cassettes (YAML files)
vcr-clean:
	rm -f tests/cassettes/*.yaml

# Fully re-record all VCR cassettes from scratch
vcr-record: vcr-clean
	pytest --vcr-record=all
