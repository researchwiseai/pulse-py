# Streamlined Workflow Validation Scripts

This directory contains validation scripts for the streamlined release workflow implementation.

## Scripts

### `test_streamlined_workflow.py`

Comprehensive validation script that tests all aspects of the streamlined release workflow:

- **Asset Count Validation**: Ensures release generates fewer than 10 assets (Requirement 1.5)
- **Compliance Document Accessibility**: Verifies all compliance documents are accessible (Requirement 2.1)
- **Release Notes Compliance**: Validates release notes contain required compliance links (Requirement 3.4)
- **Workflow Configuration**: Checks GitHub Actions workflow is properly configured

**Usage:**
```bash
# Run all validations
python scripts/test_streamlined_workflow.py

# Run with verbose output
python scripts/test_streamlined_workflow.py --verbose

# Generate detailed report
python scripts/test_streamlined_workflow.py --report-file validation_report.md
```

**Exit Codes:**
- `0`: All validations passed
- `1`: One or more validations failed

### `validate_release_notes.py`

Focused validation script that specifically tests release notes compliance:

- Validates Compliance & Verification section exists
- Checks for required compliance document links
- Verifies verification instructions are present
- Ensures installation instructions are included

**Usage:**
```bash
# Validate release notes in default workflow
python scripts/validate_release_notes.py

# Validate specific workflow file
python scripts/validate_release_notes.py --workflow-file .github/workflows/custom.yml

# Run with verbose output
python scripts/validate_release_notes.py --verbose
```

**Exit Codes:**
- `0`: Release notes are compliant
- `1`: Release notes have compliance issues

## Integration Tests

### `tests/test_streamlined_workflow_integration.py`

Pytest-based integration tests that can be run as part of the test suite:

```bash
# Run all workflow integration tests
pytest tests/test_streamlined_workflow_integration.py -v

# Run specific test
pytest tests/test_streamlined_workflow_integration.py::TestStreamlinedWorkflowIntegration::test_release_asset_count_under_limit -v
```

## Requirements Coverage

The validation scripts cover the following requirements from the streamlined release process specification:

- **Requirement 1.5**: Total number of release assets SHALL be fewer than 10 files
- **Requirement 2.1**: Release notes SHALL contain compliance document links
- **Requirement 3.4**: Release notes SHALL include required compliance sections

## Expected Asset Breakdown

The streamlined workflow should generate exactly **9 release assets**:

1. **Distribution Files (2)**:
   - `pulse_sdk-<version>-py3-none-any.whl`
   - `pulse_sdk-<version>.tar.gz`

2. **Attestation Files (2)**:
   - `pulse_sdk-<version>-py3-none-any.whl.attestation` (includes provenance)
   - `pulse_sdk-<version>.tar.gz.attestation` (includes provenance)

3. **Signature Files (4)**:
   - `pulse_sdk-<version>-py3-none-any.whl.sig`
   - `pulse_sdk-<version>-py3-none-any.whl.crt`
   - `pulse_sdk-<version>.tar.gz.sig`
   - `pulse_sdk-<version>.tar.gz.crt`

4. **Supply Chain Files (1)**:
   - `sbom.cyclonedx.json` (single SBOM format)

**Total: 9 files** (meets "fewer than 10" requirement)

## CI/CD Integration

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Validate Streamlined Workflow
  run: |
    python scripts/test_streamlined_workflow.py
    python scripts/validate_release_notes.py
```

## Troubleshooting

### Common Issues

1. **Asset count exceeds limit**: Check if workflow is generating unnecessary files
2. **Missing compliance links**: Verify release notes template in workflow file
3. **Workflow configuration issues**: Ensure workflow follows streamlined design

### Debug Mode

Run scripts with `--verbose` flag to see detailed validation steps and identify specific issues.
