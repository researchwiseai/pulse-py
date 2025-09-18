#!/usr/bin/env python3
"""
Semantic versioning validation script for Pulse SDK.

This script validates that the version in pyproject.toml follows semantic versioning
standards and can be used in CI/CD pipelines to ensure version consistency.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("ERROR: tomllib/tomli not available. Install with: pip install tomli")
        sys.exit(1)

try:
    import semver
except ImportError:
    print("ERROR: semver package not installed. Install with: pip install semver")
    sys.exit(1)


def get_version_from_pyproject() -> Optional[str]:
    """Extract version from pyproject.toml file."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"ERROR: pyproject.toml not found at {pyproject_path}")
        return None

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        version = data.get("project", {}).get("version")
        if not version:
            print("ERROR: No version found in pyproject.toml [project] section")
            return None

        return version
    except Exception as e:
        print(f"ERROR: Failed to read pyproject.toml: {e}")
        return None


def validate_semver(version: str) -> bool:
    """Validate that version follows semantic versioning."""
    try:
        # Parse the version to validate it
        parsed = semver.Version.parse(version)
        print(f"✓ Version '{version}' is valid semantic version")
        print(f"  Major: {parsed.major}")
        print(f"  Minor: {parsed.minor}")
        print(f"  Patch: {parsed.patch}")

        if parsed.prerelease:
            print(f"  Prerelease: {parsed.prerelease}")

        if parsed.build:
            print(f"  Build: {parsed.build}")

        return True
    except ValueError as e:
        print(f"ERROR: Invalid semantic version '{version}': {e}")
        return False


def main():
    """Main validation function."""
    print("Pulse SDK Version Validation")
    print("=" * 40)

    # Get version from pyproject.toml
    version = get_version_from_pyproject()
    if not version:
        sys.exit(1)

    print(f"Found version: {version}")

    # Validate semantic versioning
    if not validate_semver(version):
        sys.exit(1)

    print("\n✓ All version validations passed!")


if __name__ == "__main__":
    main()
