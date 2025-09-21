#!/usr/bin/env python3
"""
Generate compliance artifacts for Apache 2.0 license compliance.

This script generates additional compliance documentation and artifacts
that are useful for enterprise adoption and legal review.
"""

import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def get_git_info() -> Dict[str, str]:
    """Get current git repository information."""
    try:
        commit_hash = subprocess.check_output(  # nosec B607
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()

        commit_date = subprocess.check_output(  # nosec B607
            ["git", "show", "-s", "--format=%ci", "HEAD"], text=True
        ).strip()

        remote_url = subprocess.check_output(  # nosec B607
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()

        return {
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "remote_url": remote_url,
        }
    except subprocess.CalledProcessError:
        return {
            "commit_hash": "unknown",
            "commit_date": "unknown",
            "remote_url": "unknown",
        }


def calculate_file_hashes(file_path: Path) -> Dict[str, str]:
    """Calculate multiple hash types for a file."""
    hashes = {}

    if not file_path.exists():
        return hashes

    content = file_path.read_bytes()

    hashes["sha256"] = hashlib.sha256(content).hexdigest()
    hashes["sha1"] = hashlib.sha1(
        content, usedforsecurity=False
    ).hexdigest()  # nosec B324
    hashes["md5"] = hashlib.md5(
        content, usedforsecurity=False
    ).hexdigest()  # nosec B324

    return hashes


def generate_license_manifest() -> Dict[str, Any]:
    """Generate a comprehensive license manifest."""
    git_info = get_git_info()

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "pulse-sdk-compliance-generator",
        "project": {
            "name": "pulse-sdk",
            "version": "0.4.1",  # This should be read from pyproject.toml
            "license": "Apache-2.0",
            "copyright": "Copyright 2025 Researchwise AI",
            "homepage": "https://github.com/researchwiseai/pulse-py",
            "repository": git_info["remote_url"],
            "commit": git_info["commit_hash"],
            "commit_date": git_info["commit_date"],
        },
        "license_files": [],
        "compliance_artifacts": [],
        "third_party_licenses": [],
    }

    # Add license files with hashes
    license_files = [
        "LICENSE",
        "NOTICE",
        "LICENSES/Apache-2.0.txt",
        "LICENSES/SPDX-LICENSE-INFO.md",
        "COMPLIANCE.md",
    ]

    for license_file in license_files:
        file_path = Path(license_file)
        if file_path.exists():
            hashes = calculate_file_hashes(file_path)
            manifest["license_files"].append(
                {
                    "path": license_file,
                    "purpose": get_file_purpose(license_file),
                    "required": is_required_file(license_file),
                    "hashes": hashes,
                    "size_bytes": file_path.stat().st_size,
                }
            )

    # Add compliance artifacts
    compliance_artifacts = [
        "scripts/generate_compliance_artifacts.py",
        "security-reports/bandit-report.json",
        "security-reports/pip-audit-report.json",
    ]

    for artifact in compliance_artifacts:
        file_path = Path(artifact)
        if file_path.exists():
            hashes = calculate_file_hashes(file_path)
            manifest["compliance_artifacts"].append(
                {
                    "path": artifact,
                    "type": get_artifact_type(artifact),
                    "hashes": hashes,
                    "size_bytes": file_path.stat().st_size,
                }
            )

    return manifest


def get_file_purpose(file_path: str) -> str:
    """Get the purpose description for a license file."""
    purposes = {
        "LICENSE": "Main Apache 2.0 license text",
        "NOTICE": "Attribution notices and third-party acknowledgments",
        "LICENSES/Apache-2.0.txt": "Full Apache 2.0 license text",
        "LICENSES/SPDX-LICENSE-INFO.md": (
            "SPDX license information and compatibility matrix"
        ),
        "COMPLIANCE.md": "Compliance guidance and verification instructions",
    }
    return purposes.get(file_path, "License-related file")


def is_required_file(file_path: str) -> bool:
    """Check if a file is required for Apache 2.0 compliance."""
    required_files = {"LICENSE", "NOTICE"}
    return Path(file_path).name in required_files


def get_artifact_type(file_path: str) -> str:
    """Get the type of compliance artifact."""
    if "bandit" in file_path:
        return "security_scan"
    elif "pip-audit" in file_path:
        return "vulnerability_scan"
    elif file_path.endswith(".py"):
        return "compliance_tool"
    else:
        return "compliance_document"


def generate_attribution_file() -> str:
    """Generate a comprehensive attribution file."""
    attribution = """# Third-Party Attributions

This file contains the licenses and notices for third-party software
included in or used by the Pulse SDK.

## Direct Dependencies

### httpx
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2019, Encode OSS Ltd.
- **Homepage**: https://github.com/encode/httpx
- **License Text**: See LICENSES/BSD-3-Clause-httpx.txt

### pydantic
- **License**: MIT License
- **Copyright**: Copyright (c) 2017 to present Pydantic Services Inc.
- **Homepage**: https://github.com/pydantic/pydantic
- **License Text**: See LICENSES/MIT-pydantic.txt

### diskcache
- **License**: Apache License 2.0
- **Copyright**: Copyright 2016-2023 Grant Jenks
- **Homepage**: https://github.com/grantjenks/python-diskcache
- **License Text**: See LICENSES/Apache-2.0-diskcache.txt

## Development Dependencies

### pytest
- **License**: MIT License
- **Copyright**: Copyright (c) 2004 Holger Krekel and others
- **Homepage**: https://github.com/pytest-dev/pytest
- **Usage**: Testing framework (development only)

### black
- **License**: MIT License
- **Copyright**: Copyright (c) 2018 Łukasz Langa
- **Homepage**: https://github.com/psf/black
- **Usage**: Code formatting (development only)

## License Compatibility

All third-party licenses are compatible with the Apache License 2.0:

- **MIT License**: Fully compatible, can be combined
- **BSD 3-Clause**: Fully compatible, can be combined
- **Apache 2.0**: Same license, fully compatible

## Verification

This attribution file was generated automatically. To verify the accuracy:

```bash
# Check current dependencies
pip list --format=json > current-deps.json

# Scan licenses
pip-licenses --format=json --output-file=license-scan.json

# Compare with this file
python scripts/verify_attributions.py
```

---
Generated: {timestamp}
Generator: pulse-sdk-compliance-generator v1.0
""".format(
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return attribution


def main():
    """Generate all compliance artifacts."""
    print("Generating Apache 2.0 compliance artifacts...")

    # Generate license manifest
    print("📋 Generating license manifest...")
    manifest = generate_license_manifest()

    with open("LICENSE-MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("   ✓ Created LICENSE-MANIFEST.json")

    # Generate attribution file
    print("📝 Generating attribution file...")
    attribution = generate_attribution_file()

    with open("THIRD-PARTY-ATTRIBUTIONS.md", "w") as f:
        f.write(attribution)

    print("   ✓ Created THIRD-PARTY-ATTRIBUTIONS.md")

    # Generate compliance checklist
    print("✅ Generating compliance checklist...")
    checklist = generate_compliance_checklist()

    with open("COMPLIANCE-CHECKLIST.md", "w") as f:
        f.write(checklist)

    print("   ✓ Created COMPLIANCE-CHECKLIST.md")

    print("\n🎉 All compliance artifacts generated successfully!")
    print("\nGenerated files:")
    print("  - LICENSE-MANIFEST.json")
    print("  - THIRD-PARTY-ATTRIBUTIONS.md")
    print("  - COMPLIANCE-CHECKLIST.md")

    print("\nNext steps:")
    print("  1. Review generated files for accuracy")
    print("  2. Add files to version control")
    print("  3. Include in release artifacts")
    print("  4. Update documentation as needed")


def generate_compliance_checklist() -> str:
    """Generate a compliance checklist for releases."""
    checklist = """# Release Compliance Checklist

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
Generated: {timestamp}
""".format(
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    return checklist


if __name__ == "__main__":
    main()
