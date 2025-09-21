# Release Compliance Checklist

Use this checklist to ensure Apache 2.0 compliance for each release.

## Pre-Release Verification

### License Files
- [ ] `LICENSE` file contains current Apache 2.0 license text
- [ ] `NOTICE` file contains current attribution notices
- [ ] `LICENSES/Apache-2.0.txt` contains full license text
- [ ] `LICENSES/SPDX-LICENSE-INFO.md` is up to date
- [ ] `COMPLIANCE.md` contains current guidance

### Third-Party Dependencies
- [ ] All dependencies are scanned for license compatibility
- [ ] New dependencies are reviewed and approved
- [ ] `THIRD-PARTY-ATTRIBUTIONS.md` is updated
- [ ] SBOM files include all dependencies

### Security Artifacts
- [ ] Security scans are passing (Bandit, pip-audit)
- [ ] Vulnerability reports are reviewed
- [ ] No high-severity security issues remain

## Release Artifacts

### Required Files (Must be included in every release)
- [ ] Source distribution (.tar.gz)
- [ ] Wheel distribution (.whl)
- [ ] `LICENSE` file
- [ ] `NOTICE` file
- [ ] `README.md`

### Supply Chain Security (Generated automatically)
- [ ] SBOM files (SPDX and CycloneDX formats)
- [ ] Digital signatures (.sig files)
- [ ] Signing certificates (.crt files)
- [ ] Build provenance (build-provenance.json)
- [ ] GitHub attestations

### Compliance Documentation
- [ ] `LICENSE-MANIFEST.json`
- [ ] `THIRD-PARTY-ATTRIBUTIONS.md`
- [ ] `COMPLIANCE.md`
- [ ] `COMPLIANCE-CHECKLIST.md` (this file)

## Post-Release Verification

### Distribution Verification
- [ ] PyPI package includes all required files
- [ ] GitHub release includes all artifacts
- [ ] Digital signatures can be verified
- [ ] SBOM files are valid and complete

### Documentation Updates
- [ ] Release notes mention license compliance
- [ ] Documentation reflects any license changes
- [ ] Compliance guidance is current

### Legal Review (For major releases)
- [ ] Legal team has reviewed changes
- [ ] License compatibility is confirmed
- [ ] Export control requirements are met
- [ ] Trademark usage is appropriate

## Verification Commands

```bash
# Verify package contents
tar -tzf dist/pulse-sdk-*.tar.gz | grep -E "(LICENSE|NOTICE)"

# Verify signatures
cosign verify-blob --certificate *.crt --signature *.sig *.whl

# Verify SBOM
python -c "import json; print(json.load(open('sbom-wheel.spdx.json'))['name'])"

# Check license compatibility
python scripts/generate_compliance_artifacts.py
```

## Sign-off

- [ ] **Engineering Lead**: Artifacts are complete and accurate
- [ ] **Security Team**: Security requirements are met
- [ ] **Legal Team**: License compliance is verified (major releases)
- [ ] **Release Manager**: All checklist items are complete

**Release Version**: ___________
**Date**: ___________
**Signed by**: ___________

---
Generated: 2025-09-21T12:02:29.417287+00:00
