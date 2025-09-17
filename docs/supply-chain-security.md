# Supply Chain Security

The Pulse SDK implements comprehensive supply chain security measures to ensure the integrity, authenticity, and provenance of all releases. This document provides detailed information about our security practices and how to verify them.

## Overview

Our supply chain security implementation follows industry best practices and standards:

- **SLSA Build Level 3** compliance through GitHub's native attestation service
- **Software Bill of Materials (SBOM)** generation in multiple formats
- **Keyless digital signatures** using Sigstore/Cosign
- **Reproducible builds** with automated verification
- **Comprehensive provenance** tracking

## SLSA (Supply-chain Levels for Software Artifacts)

### What is SLSA?

SLSA is a security framework that helps prevent supply chain attacks by providing a common vocabulary and set of requirements for software supply chain security. Our implementation achieves **SLSA Build Level 3**.

### Our SLSA Implementation

- **Source**: All builds are tied to specific Git commits with full history
- **Build**: Builds run on GitHub-hosted runners with full isolation
- **Provenance**: Complete build provenance is generated and signed
- **Common**: All artifacts are signed and verifiable

### Build Provenance

Every release includes a `build-provenance.json` file containing:

```json
{
  "buildType": "https://github.com/researchwiseai/pulse-py/.github/workflows/publish.yml@refs/heads/main",
  "builder": {
    "id": "https://github.com/actions/runner/github-hosted"
  },
  "invocation": {
    "configSource": {
      "uri": "git+https://github.com/researchwiseai/pulse-py@<commit-sha>",
      "digest": {
        "sha1": "<commit-sha>"
      },
      "entryPoint": ".github/workflows/publish.yml"
    }
  },
  "metadata": {
    "buildInvocationId": "<run-id>-<run-attempt>",
    "reproducible": true
  }
}
```

## Software Bill of Materials (SBOM)

### What is an SBOM?

An SBOM is a comprehensive inventory of all software components, dependencies, and metadata used in building a software artifact. It provides transparency into what's included in our packages.

### Our SBOM Implementation

We generate SBOMs using [Syft](https://github.com/anchore/syft) in multiple industry-standard formats:

- **SPDX JSON** - Software Package Data Exchange format
- **CycloneDX JSON** - OWASP CycloneDX format

### SBOM Files

Each release includes the following SBOM files:

- `sbom-source.spdx.json` - SBOM for the source distribution
- `sbom-source.cyclonedx.json` - CycloneDX format for source
- `sbom-wheel.spdx.json` - SBOM for the wheel distribution
- `sbom-wheel.cyclonedx.json` - CycloneDX format for wheel

### SBOM Content

Our SBOMs include:

- All Python dependencies with exact versions
- License information for each component
- Package relationships and dependency tree
- File checksums and metadata
- Build environment information

## Digital Signatures

### Sigstore and Cosign

We use [Sigstore](https://www.sigstore.dev/) for keyless signing with [Cosign](https://github.com/sigstore/cosign). This provides:

- **Keyless signing** - No long-lived private keys to manage
- **Transparency** - All signatures are recorded in public transparency logs
- **Identity verification** - Signatures are tied to GitHub OIDC identity
- **Certificate-based verification** - Short-lived certificates for verification

### What We Sign

Every release includes signatures for:

- Source distribution (`.tar.gz`)
- Wheel distribution (`.whl`)
- All SBOM files
- Build provenance information

### Signature Files

For each signed file, we provide:

- `.sig` file - The digital signature
- `.crt` file - The signing certificate

## Verification

### Automated Verification Script

We provide a comprehensive verification script at `scripts/verify-supply-chain.sh`:

```bash
# Download and verify a specific release
./scripts/verify-supply-chain.sh v1.0.0
```

The script will:

1. Download all release artifacts
2. Verify digital signatures using Cosign
3. Validate SBOM content (if Syft is installed)
4. Display a comprehensive verification report

### Manual Verification

#### Prerequisites

Install Cosign:

```bash
# Linux
curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign

# macOS with Homebrew
brew install cosign
```

#### Verify Package Signatures

```bash
# Verify wheel package
cosign verify-blob \
  --certificate pulse_sdk-1.0.0-py3-none-any.whl.crt \
  --signature pulse_sdk-1.0.0-py3-none-any.whl.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse_sdk-1.0.0-py3-none-any.whl

# Verify source distribution
cosign verify-blob \
  --certificate pulse-sdk-1.0.0.tar.gz.crt \
  --signature pulse-sdk-1.0.0.tar.gz.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  pulse-sdk-1.0.0.tar.gz
```

#### Verify SBOM Signatures

```bash
cosign verify-blob \
  --certificate sbom-wheel.spdx.json.crt \
  --signature sbom-wheel.spdx.json.sig \
  --certificate-identity-regexp "https://github.com/researchwiseai/pulse-py/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  sbom-wheel.spdx.json
```

## Reproducible Builds

### What are Reproducible Builds?

Reproducible builds ensure that the same source code always produces identical binary artifacts, making it possible to verify that published binaries match the source code.

### Our Implementation

- **Standardized environment** - Fixed Python version and build tools
- **Deterministic timestamps** - Using `SOURCE_DATE_EPOCH` from Git
- **Consistent dependencies** - Pinned build tool versions
- **Automated verification** - Separate workflow verifies reproducibility

### Verification Workflow

Our reproducible build verification workflow:

1. Checks out the exact commit used for the original build
2. Sets up an identical build environment
3. Rebuilds the packages
4. Compares checksums with the original build
5. Reports any differences

### Manual Reproducibility Verification

To verify a build locally:

```bash
# Checkout the release tag
git checkout v1.0.0

# Set reproducible build environment
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
export PYTHONHASHSEED=0

# Build
python -m build --sdist --wheel

# Compare checksums with published artifacts
sha256sum dist/*
```

## Security Monitoring

### Continuous Monitoring

Our security monitoring includes:

- **Daily dependency scans** - Automated vulnerability scanning
- **SBOM analysis** - Regular analysis of dependency changes
- **Signature verification** - Automated verification of all releases
- **Reproducibility checks** - Regular verification of build reproducibility

### Incident Response

In case of a security incident:

1. **Detection** - Automated scanning or external report
2. **Assessment** - Evaluate impact and severity
3. **Response** - Patch, rebuild, and re-release if necessary
4. **Communication** - Notify users through GitHub Security Advisories
5. **Post-incident** - Review and improve security measures

## Compliance and Standards

Our supply chain security implementation helps ensure compliance with:

- **SLSA Framework** - Build Level 3 compliance
- **NIST SSDF** - Secure Software Development Framework
- **Executive Order 14028** - Software supply chain security requirements
- **OWASP SCVS** - Software Component Verification Standard

## Integration with Package Managers

### PyPI Integration

- **Trusted publishing** - Using OIDC for secure PyPI publishing
- **Attestations** - PyPI attestations are automatically generated
- **Metadata** - Rich package metadata for better discoverability

### Enterprise Integration

For enterprise users, our supply chain security features enable:

- **Policy enforcement** - Verify signatures before installation
- **Dependency tracking** - Use SBOMs for license and vulnerability management
- **Audit trails** - Complete provenance for compliance requirements

## Best Practices for Users

### For Individual Developers

1. **Verify signatures** - Use our verification script for critical deployments
2. **Review SBOMs** - Check dependencies for security and licensing concerns
3. **Pin versions** - Use exact version pins for reproducible deployments

### For Organizations

1. **Implement verification** - Integrate signature verification into CI/CD
2. **SBOM analysis** - Use SBOMs for vulnerability and license management
3. **Policy enforcement** - Require signed packages in production
4. **Audit compliance** - Use provenance information for compliance reporting

## Troubleshooting

### Common Issues

#### Signature Verification Fails

- Ensure you're using the correct certificate identity and OIDC issuer
- Check that you have the latest version of Cosign
- Verify you're using the correct signature and certificate files

#### SBOM Content Differences

- Different versions of Syft may produce slightly different SBOMs
- This is normal and doesn't indicate a security issue
- Focus on major component differences, not metadata variations

#### Reproducibility Verification Fails

- Ensure you're using the exact same Python version and build tools
- Check that `SOURCE_DATE_EPOCH` is set correctly
- Minor differences in metadata are acceptable

### Getting Help

If you encounter issues with supply chain security verification:

1. Check our [troubleshooting guide](../README.md#troubleshooting)
2. Review the [security policy](../SECURITY.md)
3. Open an issue on [GitHub](https://github.com/researchwiseai/pulse-py/issues)
4. For security-related issues, email dev@researchwiseai.com

## Future Enhancements

We're continuously improving our supply chain security:

- **Enhanced SBOM analysis** - Automated vulnerability scanning of SBOMs
- **Multi-platform signatures** - Support for different architectures
- **Integration testing** - Automated testing of verification workflows
- **Policy templates** - Pre-built policies for common use cases

---

For more information about our security practices, see our [Security Policy](../SECURITY.md).
