# Security Scanning Scripts

This directory contains scripts for running security scans on the Pulse SDK.

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