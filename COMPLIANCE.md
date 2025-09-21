# Apache 2.0 License Compliance Guide

This document provides guidance for ensuring compliance with the Apache License 2.0 when using or distributing the Pulse SDK.

## Required Files for Distribution

When distributing the Pulse SDK or derivative works, you must include:

### ✅ Core License Files
- [ ] `LICENSE` - The main Apache 2.0 license text
- [ ] `NOTICE` - Attribution notices and third-party acknowledgments
- [ ] `LICENSES/Apache-2.0.txt` - Full Apache 2.0 license text
- [ ] `LICENSES/SPDX-LICENSE-INFO.md` - SPDX license information

### ✅ Supply Chain Security Artifacts (Available in GitHub Releases)
- [ ] `sbom-source.spdx.json` - Software Bill of Materials (SPDX format) for source
- [ ] `sbom-wheel.spdx.json` - Software Bill of Materials (SPDX format) for wheel
- [ ] `sbom-source.cyclonedx.json` - Software Bill of Materials (CycloneDX format) for source
- [ ] `sbom-wheel.cyclonedx.json` - Software Bill of Materials (CycloneDX format) for wheel
- [ ] `build-provenance.json` - SLSA build provenance information
- [ ] `*.sig` files - Cryptographic signatures for all artifacts
- [ ] `*.crt` files - Signing certificates for verification

## Apache 2.0 Compliance Requirements

### 1. License Notice Preservation
You must retain all copyright, patent, trademark, and attribution notices from the original work.

### 2. Modified File Notices
If you modify any files, you must add prominent notices stating that you changed the files.

### 3. NOTICE File Distribution
If the original work includes a NOTICE file, you must include it in your distribution.

### 4. License File Distribution
You must provide a copy of the Apache 2.0 license with your distribution.

## Enterprise Compliance Checklist

### Legal Review
- [ ] Legal team has reviewed Apache 2.0 license terms
- [ ] Patent implications have been assessed
- [ ] Trademark usage guidelines are understood
- [ ] Export control requirements have been evaluated

### Technical Implementation
- [ ] All required files are included in distribution
- [ ] SBOM files are reviewed for dependency compliance
- [ ] Digital signatures are verified before use
- [ ] Build provenance is validated

### Documentation
- [ ] Internal compliance documentation is updated
- [ ] Third-party license obligations are documented
- [ ] Attribution requirements are met
- [ ] Modification notices are added where applicable

## Verification Commands

### Verify Package Integrity
```bash
# Download release artifacts from GitHub
# Verify wheel signature
cosign verify-blob \
  --certificate pulse_sdk-*.whl.crt \
  --signature pulse_sdk-*.whl.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse_sdk-*.whl

# Verify source distribution signature
cosign verify-blob \
  --certificate pulse-sdk-*.tar.gz.crt \
  --signature pulse-sdk-*.tar.gz.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse-sdk-*.tar.gz
```

### Verify SBOM Integrity
```bash
# Verify SBOM signatures
cosign verify-blob \
  --certificate sbom-wheel.spdx.json.crt \
  --signature sbom-wheel.spdx.json.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  sbom-wheel.spdx.json
```

### License Scanning
```bash
# Install license scanning tools
pip install pip-licenses licensecheck

# Scan installed dependencies
pip-licenses --format=json --output-file=license-report.json

# Check for license compatibility
licensecheck --format=json pulse-sdk
```

## Common Compliance Scenarios

### Scenario 1: Using Pulse SDK in Commercial Software
- ✅ **Permitted**: Apache 2.0 allows commercial use
- ✅ **Requirements**: Include LICENSE and NOTICE files
- ✅ **Patent Grant**: Automatic patent license included
- ❌ **Restrictions**: Cannot use Researchwise AI trademarks without permission

### Scenario 2: Modifying Pulse SDK Source Code
- ✅ **Permitted**: Apache 2.0 allows modifications
- ✅ **Requirements**: Add modification notices to changed files
- ✅ **Distribution**: Must include original LICENSE and NOTICE
- ✅ **Additional**: Can add your own license terms for modifications

### Scenario 3: Redistributing Pulse SDK
- ✅ **Permitted**: Apache 2.0 allows redistribution
- ✅ **Requirements**: Include all original license files
- ✅ **Format**: Can redistribute in source or binary form
- ✅ **Attribution**: Must preserve all attribution notices

### Scenario 4: Creating Derivative Works
- ✅ **Permitted**: Apache 2.0 allows derivative works
- ✅ **Requirements**: Include original LICENSE and NOTICE
- ✅ **Modifications**: Add notices for any changes made
- ✅ **Licensing**: Can license derivative work under different terms

## Risk Assessment

### Low Risk ✅
- Using Pulse SDK as-is in commercial applications
- Redistributing unmodified packages with proper attribution
- Creating applications that depend on Pulse SDK

### Medium Risk ⚠️
- Modifying Pulse SDK source code (requires change notices)
- Combining with GPL-licensed code (check compatibility)
- Using in patent-sensitive environments (review patent clauses)

### High Risk ❌
- Removing or modifying license notices
- Using Researchwise AI trademarks without permission
- Combining with GPL-2.0 licensed code (incompatible)

## Support and Resources

### Legal Questions
- **Email**: legal@researchwiseai.com
- **Response Time**: 2-3 business days
- **Scope**: License interpretation, compliance guidance

### Technical Questions
- **GitHub Issues**: https://github.com/researchwiseai/pulse-py/issues
- **Email**: support@researchwiseai.com
- **Documentation**: https://researchwiseai.github.io/pulse-py/

### External Resources
- [Apache License 2.0 FAQ](https://www.apache.org/foundation/license-faq.html)
- [SPDX License List](https://spdx.org/licenses/)
- [OpenChain Project](https://www.openchainproject.org/)
- [SLSA Framework](https://slsa.dev/)

---

**Document Version**: 1.0
**Last Updated**: January 2025
**Next Review**: April 2025
**License**: Apache-2.0
