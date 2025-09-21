# Security Policy

## Supported Versions

We provide security updates for the following versions of Pulse SDK:

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Pulse SDK, please report it responsibly.

### How to Report

**For security-related issues, please email:** dev@researchwiseai.com

**Do not** create public GitHub issues for security vulnerabilities.

### What to Include

When reporting a security vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes or mitigations

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with our assessment
- **Resolution**: Security fixes will be prioritized and released as soon as possible

### Security Best Practices

When using Pulse SDK:

1. **Keep Dependencies Updated**: Regularly update to the latest version
2. **Secure Credentials**: Never commit API credentials to version control
3. **Environment Variables**: Use environment variables for sensitive configuration
4. **Network Security**: Ensure secure network connections when using the API
5. **Data Handling**: Follow your organization's data handling policies

### Supply Chain Security

Pulse SDK implements several supply chain security measures:

- **SBOM Generation**: Software Bill of Materials (SBOM) in CycloneDX format
- **Code Signing**: All releases are cryptographically signed
- **Build Provenance**: Detailed build information for verification
- **Dependency Scanning**: Regular security scans of dependencies

For detailed supply chain security information, see [Supply Chain Security](docs/supply-chain-security.md).

### Security Contacts

- **General Security Questions**: dev@researchwiseai.com
- **Vulnerability Reports**: dev@researchwiseai.com
- **Security Documentation**: [COMPLIANCE.md](COMPLIANCE.md)

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities and helping us maintain the security of Pulse SDK.
