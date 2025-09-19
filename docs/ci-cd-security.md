# CI/CD Security and Quality Assurance

This document describes the comprehensive security and quality assurance measures implemented in the Pulse SDK CI/CD pipeline.

## Overview

The CI/CD pipeline includes multiple layers of security and quality checks to ensure code safety, dependency security, and maintainability:

- **Static Application Security Testing (SAST)** with Bandit
- **Dependency Vulnerability Scanning** with pip-audit
- **Code Coverage Enforcement** (90% minimum)
- **Conventional Commit Validation**
- **Secret Scanning** (basic implementation)
- **Code Quality Checks** (formatting, linting)

## Security Scanning

### Bandit SAST Scanning

**Purpose**: Identifies common security issues in Python code through static analysis.

**Configuration**: Configured via `pyproject.toml` with appropriate exclusions for false positives.

**Triggers**:
- Every pull request (medium/high severity threshold)
- Every push to main branch
- Weekly scheduled comprehensive scan
- Manual workflow dispatch

**Failure Conditions**:
- High or medium severity security issues in CI
- Any severity issues in comprehensive security scan

**Reports**:
- JSON format for automated processing
- SARIF format uploaded to GitHub Security tab
- Artifacts retained for 90 days

### Dependency Vulnerability Scanning

**Purpose**: Scans Python dependencies for known security vulnerabilities.

**Tool**: pip-audit using the Python Packaging Advisory Database

**Triggers**:
- Every CI run (pull requests and main branch)
- Weekly scheduled comprehensive scan
- Manual workflow dispatch

**Failure Conditions**:
- Critical or high severity vulnerabilities
- Vulnerabilities with available fixes

**Features**:
- Scans installed packages and requirements files
- Provides detailed vulnerability descriptions
- Suggests fix versions when available

### Secret Scanning

**Purpose**: Detects potential secrets and credentials in code and commit messages.

**Implementation**: Basic pattern matching for common secret patterns

**Checks**:
- Commit messages for sensitive keywords
- Code files for hardcoded credentials
- Common secret patterns (API keys, passwords, tokens)

**Note**: This is a basic implementation. For production use, consider tools like:
- GitHub's native secret scanning
- TruffleHog
- GitLeaks

## Code Quality Assurance

### Test Coverage

**Minimum Threshold**: 90%

**Enforcement**:
- CI pipeline fails if coverage drops below 90%
- Coverage reports posted as PR comments
- HTML and JSON reports generated

**Configuration**: Defined in `pyproject.toml` under `[tool.coverage.*]`

**Exclusions**:
- Test files
- Example scripts
- Documentation
- Cache directories

### Code Formatting and Linting

**Tools**:
- **Black**: Code formatting (line length 88)
- **Ruff**: Fast Python linter
- **nbqa**: Jupyter notebook formatting

**Enforcement**:
- Pre-commit hooks prevent improperly formatted commits
- CI pipeline validates formatting
- Automatic fixes available via `make reformat`

### Conventional Commits

**Purpose**: Ensures consistent commit message format for automated changelog generation.

**Format**: `type(scope): description`

**Supported Types**:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions/modifications
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `revert`: Commit reverts

**Validation**:
- Pre-commit hook validates commit messages
- CI validates PR titles for conventional format
- Automated changelog generation based on commit types

## Branch Protection

### Required Status Checks

The following checks must pass before merging to main:

1. **CI Pipeline** (`test` job)
   - Code formatting validation
   - Linting checks
   - Security scanning (Bandit)
   - Vulnerability scanning (pip-audit)
   - Test execution with 90% coverage
   - Secret scanning
   - All quality gates validation

2. **Conventional Commit Validation**
   - PR title follows conventional commit format
   - Commit messages validated via pre-commit

### Branch Protection Rules

- **Require status checks**: All CI checks must pass
- **Require up-to-date branches**: Branches must be current with main
- **Require pull request reviews**: At least 1 approving review
- **Dismiss stale reviews**: When new commits are pushed
- **Require conversation resolution**: All PR conversations must be resolved
- **Restrict force pushes**: Prevent force pushes to main
- **Restrict deletions**: Prevent branch deletion

## Pre-commit Hooks

Pre-commit hooks run locally before each commit to catch issues early:

```yaml
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

**Configured Hooks**:
1. **Basic checks**: trailing whitespace, end-of-file-fixer, YAML validation
2. **Code formatting**: Black for Python, nbqa for notebooks
3. **Linting**: Ruff with auto-fix
4. **Security**: Bandit SAST scanning
5. **Dependencies**: pip-audit vulnerability scanning
6. **Commit messages**: Conventional commit validation

## Workflow Files

### `.github/workflows/ci.yml`
Main CI pipeline with comprehensive quality and security checks.

### `.github/workflows/security.yml`
Dedicated security scanning with detailed reporting and SARIF upload.

### `.github/workflows/branch-protection.yml`
Automated branch protection rule setup (requires admin permissions).

## Security Reports and Artifacts

### Artifact Retention
- **Security reports**: 90 days
- **Coverage reports**: 30 days
- **Build artifacts**: 30 days

### Report Formats
- **Bandit**: JSON and SARIF formats
- **pip-audit**: JSON and columns formats
- **Coverage**: HTML, JSON, and term-missing formats

### GitHub Security Integration
- SARIF reports uploaded to GitHub Security tab
- Security advisories for dependency vulnerabilities
- Automated security updates via Dependabot

## Monitoring and Alerts

### Scheduled Scans
- **Weekly comprehensive security scan**: Every Monday at 2 AM UTC
- **Dependency updates**: Automated via Dependabot
- **Security advisories**: GitHub native notifications

### Failure Notifications
- CI failures notify via GitHub status checks
- Security issues create GitHub Security alerts
- Coverage drops trigger PR comments

## Best Practices

### For Developers

1. **Install pre-commit hooks**: `pre-commit install`
2. **Run tests locally**: `make test` or `pytest`
3. **Check coverage**: `make coverage` or `pytest --cov`
4. **Format code**: `make format` or `black .`
5. **Lint code**: `make lint` or `ruff check`
6. **Security scan**: `bandit -r pulse`
7. **Vulnerability check**: `pip-audit`

### For Security

1. **Review security reports** in GitHub Security tab
2. **Monitor dependency alerts** and update promptly
3. **Validate security configurations** in pyproject.toml
4. **Review and update exclusions** as needed
5. **Test security tools** with known vulnerable code

### For Maintenance

1. **Update tool versions** in workflows and pre-commit config
2. **Review and adjust thresholds** based on project needs
3. **Monitor CI performance** and optimize as needed
4. **Update documentation** when adding new checks
5. **Test branch protection rules** after repository changes

## Troubleshooting

### Common Issues

**Coverage Below Threshold**:
```bash
# Check coverage locally
pytest --cov=pulse --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

**Security Scan Failures**:
```bash
# Run Bandit locally
bandit -r pulse -f json
# Review and fix issues or add exclusions if false positives
```

**Dependency Vulnerabilities**:
```bash
# Check vulnerabilities
pip-audit --desc
# Update dependencies
pip install --upgrade package-name
```

**Pre-commit Hook Failures**:
```bash
# Run specific hook
pre-commit run bandit --all-files
# Skip hooks if needed (not recommended)
git commit --no-verify
```

### Getting Help

- **CI Issues**: Check workflow logs in GitHub Actions
- **Security Questions**: Review SECURITY.md
- **Coverage Issues**: See docs/debugging.md
- **Tool Documentation**: Links in pyproject.toml comments
