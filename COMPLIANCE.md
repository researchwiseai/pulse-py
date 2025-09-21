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
- [ ] `sbom.cyclonedx.json` - Software Bill of Materials (CycloneDX format) covering both source and wheel distributions
- [ ] `build-provenance.json` - SLSA build provenance information
- [ ] `*.sig` files - Cryptographic signatures for distribution artifacts only
- [ ] `*.crt` files - Signing certificates for distribution verification
- [ ] `*.attestation` files - Build attestations for distribution artifacts

### ✅ Compliance Documents (Available in Repository)
All compliance documents are maintained in the repository root and accessible via direct links:
- [ ] [LICENSE](https://github.com/researchwiseai/pulse-py/blob/main/LICENSE) - The main Apache 2.0 license text
- [ ] [NOTICE](https://github.com/researchwiseai/pulse-py/blob/main/NOTICE) - Attribution notices and third-party acknowledgments
- [ ] [COMPLIANCE.md](https://github.com/researchwiseai/pulse-py/blob/main/COMPLIANCE.md) - This compliance guide
- [ ] [SECURITY.md](https://github.com/researchwiseai/pulse-py/blob/main/SECURITY.md) - Security policy and reporting procedures
- [ ] [THIRD-PARTY-ATTRIBUTIONS.md](https://github.com/researchwiseai/pulse-py/blob/main/THIRD-PARTY-ATTRIBUTIONS.md) - Third-party license attributions

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
- [ ] Single CycloneDX SBOM file is reviewed for dependency compliance
- [ ] Digital signatures are verified for distribution artifacts only
- [ ] Build provenance is validated
- [ ] Compliance documents are accessible via repository links

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
  --certificate pulse_sdk-{version}-py3-none-any.whl.crt \
  --signature pulse_sdk-{version}-py3-none-any.whl.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse_sdk-{version}-py3-none-any.whl

# Verify source distribution signature
cosign verify-blob \
  --certificate pulse_sdk-{version}.tar.gz.crt \
  --signature pulse_sdk-{version}.tar.gz.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse_sdk-{version}.tar.gz
```

### Verify SBOM and Build Provenance
```bash
# Validate SBOM format (CycloneDX)
python -c "import json; sbom = json.load(open('sbom.cyclonedx.json')); print(f'SBOM Format: {sbom[\"bomFormat\"]} v{sbom[\"specVersion\"]}')"

# Verify build provenance exists
test -f build-provenance.json && echo "Build provenance available" || echo "Build provenance missing"

# Check SBOM completeness
python -c "
import json
sbom = json.load(open('sbom.cyclonedx.json'))
components = sbom.get('components', [])
print(f'SBOM contains {len(components)} components')
for comp in components[:5]:  # Show first 5 components
    print(f'  - {comp[\"name\"]} v{comp.get(\"version\", \"unknown\")}')
"
```

### License Scanning
```bash
# Install license scanning tools
pip install pip-licenses licensecheck

# Scan installed dependencies
pip-licenses --format=json --output-file=license-report.json

# Check for license compatibility
licensecheck --format=json pulse-sdk

# Extract license information from CycloneDX SBOM
python -c "
import json
sbom = json.load(open('sbom.cyclonedx.json'))
print('License Information from SBOM:')
for comp in sbom.get('components', []):
    licenses = comp.get('licenses', [])
    if licenses:
        license_names = [lic.get('license', {}).get('id', 'Unknown') for lic in licenses]
        print(f'  {comp[\"name\"]}: {license_names}')
"
```

## Accessing Compliance Documents

### Repository-Based Access
All compliance documents are maintained in the repository root and can be accessed directly:

- **Latest Version**: Use `main` branch links for the most current compliance information
- **Release-Specific**: Use release tag links (e.g., `v0.4.1`) for version-specific compliance
- **Permanent Links**: GitHub blob URLs provide stable access to specific document versions

### Quick Access Links
```bash
# Download compliance documents for offline review
curl -O https://raw.githubusercontent.com/researchwiseai/pulse-py/main/LICENSE
curl -O https://raw.githubusercontent.com/researchwiseai/pulse-py/main/NOTICE
curl -O https://raw.githubusercontent.com/researchwiseai/pulse-py/main/COMPLIANCE.md
curl -O https://raw.githubusercontent.com/researchwiseai/pulse-py/main/SECURITY.md
curl -O https://raw.githubusercontent.com/researchwiseai/pulse-py/main/THIRD-PARTY-ATTRIBUTIONS.md
```

### Supply Chain Security Artifacts
Release artifacts are streamlined to include only essential files:
- **Distribution Files**: `pulse_sdk-{version}.whl`, `pulse_sdk-{version}.tar.gz`
- **Signatures**: `.sig`, `.crt`, `.attestation` files for distributions only
- **SBOM**: Single `sbom.cyclonedx.json` file covering both distributions
- **Provenance**: `build-provenance.json` for SLSA compliance

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

**Document Version**: 1.1
**Last Updated**: September 2025
**Next Review**: December 2025
**License**: Apache-2.0
**Changes**: Updated for streamlined release process with single CycloneDX SBOM format and repository-based compliance document access
