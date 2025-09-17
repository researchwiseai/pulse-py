# Scripts

This directory contains utility scripts for the Pulse SDK project.

## Available Scripts

- `security-scan.sh` - Security scanning script for CI/CD pipelines
- `verify-supply-chain.sh` - Supply chain security verification script
- `conventional-commits-guide.md` - Guide for writing conventional commit messages

## security-scan.sh

Runs comprehensive security scans including:

- **Bandit SAST**: Static Application Security Testing for Python code
- **pip-audit**: Dependency vulnerability scanning

### Usage

```bash
# Make sure you have the security tools installed
pip install bandit pip-audit

# Run the security scan
./scripts/security-scan.sh
```

### Output

The script generates reports in the `security-reports/` directory:

- `bandit-report.json`: Detailed Bandit findings in JSON format
- `bandit-report.txt`: Human-readable Bandit report
- `pip-audit-report.json`: Dependency vulnerability report

### Configuration

The security scan is configured to skip certain false positives that are acceptable for this codebase:

- **B101** (assert_used): Used for runtime validation in client code
- **B110** (try_except_pass): Used for cleanup operations
- **B105** (hardcoded_password_string): These are public OAuth endpoints
- **B311** (random): Used for sampling, not cryptographic purposes
- **B403** (pickle import): Used for cache key generation only
- **B601** (shell=True): May be needed for subprocess operations

### CI Integration

These same security scans run automatically in the CI pipeline:

- On every push and pull request (`.github/workflows/ci.yml`)
- Weekly scheduled scans (`.github/workflows/security.yml`)
- As pre-commit hooks (`.pre-commit-config.yaml`)

### Troubleshooting

If you encounter issues:

1. Ensure you have the required tools installed: `pip install bandit pip-audit`
2. Check that you're running from the project root directory
3. Verify Python 3.8+ is available
4. For permission issues, run: `chmod +x scripts/security-scan.sh`
#
# verify-supply-chain.sh

Provides comprehensive verification of Pulse SDK releases including digital signatures, SBOM integrity, and build provenance.

### Usage

```bash
# Verify a specific release
./scripts/verify-supply-chain.sh v1.0.0
```

### What it verifies

- **Digital signatures** - Verifies Sigstore/Cosign signatures for all artifacts
- **SBOM integrity** - Validates Software Bill of Materials content
- **Build provenance** - Checks build provenance information
- **Certificate chains** - Verifies signing certificates against GitHub OIDC

### Prerequisites

The script will automatically install required tools:
- `cosign` - For signature verification
- `syft` - For SBOM content verification (optional)
- `jq` - For JSON processing (optional, for detailed reports)

### Output

The script provides:
- Step-by-step verification progress
- Detailed verification report
- File checksums and metadata
- SBOM and provenance summaries
- Cleanup of temporary files

### Example Output

```
================================================
  Pulse SDK Supply Chain Security Verification
================================================

[STEP] Checking dependencies...
[SUCCESS] Dependencies checked
[STEP] Downloading release artifacts for version v1.0.0...
[INFO] Downloading from: https://github.com/researchwiseai/pulse-py/releases/download/v1.0.0
[STEP] Verifying digital signatures...
[INFO] Verifying signature for pulse_sdk-1.0.0-py3-none-any.whl
[SUCCESS] Wheel signature verified
[SUCCESS] Source distribution signature verified
[SUCCESS] SBOM signature verified
[STEP] Verifying SBOM content...
[SUCCESS] SBOM content verification passed
[SUCCESS] Supply chain verification completed successfully!
```
