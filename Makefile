## Makefile for running tests and managing VCR cassettes

.PHONY: clean build test fmt lint vcr-clean vcr-record docs-test docs-build docs-serve

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

# Format code using black and nbqa (only source directories)
fmt:
	black pulse tests examples scripts
	nbqa black examples

# Lint code using ruff
lint:
	ruff check pulse tests

# Run security scans locally
security:
	@echo "Running local security scans..."
	bandit -r pulse --format json --output bandit-report.json --severity-level medium --confidence-level medium
	pip-audit --format=json --output=pip-audit-report.json --desc
	python scripts/security_config.py
	@echo "✅ Security scans completed"

# Remove all recorded VCR cassettes (YAML files)
vcr-clean:
	rm -f tests/cassettes/*.yaml

# Fully re-record all VCR cassettes from scratch
vcr-record: vcr-clean
	pytest --vcr-record=all

# Run documentation tests
docs-test:
	@echo "Running documentation validation tests..."
	python scripts/run_doctests.py --verbose
	python scripts/validate_quickstart.py --verbose
	python scripts/check_links.py --no-external --verbose
	python scripts/validate_docs.py --build-docs --verbose
	@echo "✅ All documentation tests passed"

# Build documentation
docs-build:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Show release information and process
release-info:
	@echo "🚀 Release Process Information"
	@echo "=============================="
	@echo ""
	@echo "This project uses automated releases via Release Please."
	@echo "Manual version bumping is NOT needed."
	@echo ""
	@echo "📝 To prepare for a release:"
	@echo "  1. Use conventional commit messages:"
	@echo "     - feat: for new features (minor version bump)"
	@echo "     - fix: for bug fixes (patch version bump)"
	@echo "     - feat!: or fix!: for breaking changes (major version bump)"
	@echo "     - docs:, refactor:, perf:, etc. for other changes"
	@echo ""
	@echo "  2. Push commits to main branch"
	@echo "  3. Release Please will automatically create a release PR"
	@echo "  4. Review and merge the release PR to trigger the release"
	@echo ""
	@echo "🔍 Current version: $(shell grep -E '^version = ' pyproject.toml | sed -E 's/version = \"([^\"]+)\"/\1/')"
	@echo "📋 Recent commits:"
	@git log --oneline -5
	@echo ""
	@echo "🔗 For more info: https://github.com/googleapis/release-please"
