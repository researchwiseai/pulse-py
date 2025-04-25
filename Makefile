## Makefile for running tests and managing VCR cassettes

.PHONY: test vcr-clean vcr-record

# Run all tests normally (uses record_mode='once' for VCR)
test:
	pytest

# Remove all recorded VCR cassettes (YAML files)
vcr-clean:
	rm -f tests/cassettes/*.yaml

# Fully re-record all VCR cassettes from scratch
vcr-record: vcr-clean
	pytest --vcr-record=all
