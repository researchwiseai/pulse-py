# Security Policy

## Automated Security Scanning

This project implements comprehensive security scanning as part of our CI/CD pipeline:

### Static Application Security Testing (SAST)

- **Tool**: [Bandit](https://bandit.readthedocs.io/)
- **Scope**: All Python source code in the `pulse/` directory
- **Configuration**: See `[tool.bandit]` section in `pyproject.toml`
- **Exclusions**: Test files, examples, and documentation
- **Severity Threshold**: Medium and above
- **Integration**: 
  - Pre-commit hooks for immediate feedback
  - CI pipeline with SARIF upload to GitHub Security tab
  - Weekly scheduled scans

### Dependency Vulnerability Scanning

- **Tool**: [pip-audit](https://pypi.org/project/pip-audit/)
- **Scope**: All production and development dependencies
- **Configuration**: See `.pip-audit.toml`
- **Severity Threshold**: Medium and above
- **Integration**:
  - CI pipeline on every push and PR
  - Weekly scheduled scans
  - JSON reports for automated processing

### Secret Scanning

- **Tool**: GitHub native secret scanning
- **Scope**: All repository content and history
- **Integration**: Automatic scanning with GitHub Security tab alerts

## Security Scan Results

Security scan results are available in multiple formats:

1. **GitHub Security Tab**: SARIF results from Bandit are automatically uploaded
2. **CI Artifacts**: JSON and SARIF reports are stored as workflow artifacts
3. **Pre-commit Output**: Immediate feedback during development

## Handling Security Issues

### Critical/High Severity Issues

- CI pipeline will fail for high severity Bandit findings
- Critical/high severity dependency vulnerabilities will fail the security workflow
- All critical issues must be resolved before merging to main branch

### Medium/Low Severity Issues

- Medium severity issues are reported but don't fail the build
- Low severity issues are informational only
- Regular review and remediation is recommended

### False Positives

If a security finding is determined to be a false positive:

1. Add the specific issue ID to the `ignore-vulns` list in `.pip-audit.toml` (for dependency issues)
2. Add appropriate exclusions to the `[tool.bandit]` configuration in `pyproject.toml` (for SAST issues)
3. Document the reasoning in commit messages

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please report it privately by emailing dev@researchwiseai.com. Do not create a public GitHub issue for security vulnerabilities.

## Security Best Practices

This project follows these security best practices:

- Regular dependency updates
- Automated vulnerability scanning
- Secure coding practices validated by SAST
- Supply chain security with SLSA attestations
- Minimal privilege principles in CI/CD

## Compliance

Our security scanning helps ensure compliance with:

- OWASP Top 10 security risks
- Common Weakness Enumeration (CWE) standards
- Software supply chain security best practices