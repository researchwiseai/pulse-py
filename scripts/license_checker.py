#!/usr/bin/env python3
"""
License Compatibility Checker for Pulse SDK

This script checks all dependencies for license compatibility and generates
SPDX license information for compliance reporting.
"""

import json
import sys
from typing import Dict
import pkg_resources
import requests


# SPDX License compatibility matrix
# Based on: https://spdx.org/licenses/
COMPATIBLE_LICENSES = {
    "MIT": {
        "compatible_with": [
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache-2.0",
            "ISC",
            "Unlicense",
        ],
        "risk_level": "low",
        "commercial_use": True,
        "copyleft": False,
    },
    "Apache-2.0": {
        "compatible_with": ["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"],
        "risk_level": "low",
        "commercial_use": True,
        "copyleft": False,
    },
    "BSD-2-Clause": {
        "compatible_with": ["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"],
        "risk_level": "low",
        "commercial_use": True,
        "copyleft": False,
    },
    "BSD-3-Clause": {
        "compatible_with": ["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"],
        "risk_level": "low",
        "commercial_use": True,
        "copyleft": False,
    },
    "ISC": {
        "compatible_with": ["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC"],
        "risk_level": "low",
        "commercial_use": True,
        "copyleft": False,
    },
    "GPL-2.0": {
        "compatible_with": ["GPL-2.0", "GPL-3.0"],
        "risk_level": "high",
        "commercial_use": True,
        "copyleft": True,
    },
    "GPL-3.0": {
        "compatible_with": ["GPL-3.0"],
        "risk_level": "high",
        "commercial_use": True,
        "copyleft": True,
    },
    "LGPL-2.1": {
        "compatible_with": [
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache-2.0",
            "LGPL-2.1",
            "GPL-2.0",
            "GPL-3.0",
        ],
        "risk_level": "medium",
        "commercial_use": True,
        "copyleft": True,
    },
    "LGPL-3.0": {
        "compatible_with": [
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache-2.0",
            "LGPL-3.0",
            "GPL-3.0",
        ],
        "risk_level": "medium",
        "commercial_use": True,
        "copyleft": True,
    },
    "MPL-2.0": {
        "compatible_with": [
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache-2.0",
            "MPL-2.0",
        ],
        "risk_level": "medium",
        "commercial_use": True,
        "copyleft": True,
    },
}

# Our project license (MIT)
PROJECT_LICENSE = "MIT"


class LicenseChecker:
    """Check license compatibility for all dependencies."""

    def __init__(self):
        self.dependencies = {}
        self.license_cache = {}
        self.compatibility_issues = []

    def get_installed_packages(self) -> Dict[str, str]:
        """Get all installed packages and their versions."""
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name] = dist.version
        return packages

    def get_package_license_from_pypi(self, package_name: str) -> str:
        """Fetch license information from PyPI API."""
        if package_name in self.license_cache:
            return self.license_cache[package_name]

        try:
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            license_info = data.get("info", {}).get("license", "Unknown")

            # Try to get from classifiers if license field is empty
            if not license_info or license_info.lower() in ["unknown", "", "none"]:
                classifiers = data.get("info", {}).get("classifiers", [])
                license_classifiers = [
                    c for c in classifiers if c.startswith("License ::")
                ]
                if license_classifiers:
                    # Extract license from classifier
                    license_info = license_classifiers[0].split(" :: ")[-1]

            self.license_cache[package_name] = license_info
            return license_info

        except Exception as e:
            print(f"Warning: Could not fetch license for {package_name}: {e}")
            return "Unknown"

    def normalize_license_name(self, license_str: str) -> str:
        """Normalize license names to SPDX identifiers."""
        if not license_str or license_str.lower() in ["unknown", "", "none"]:
            return "Unknown"

        # Common license name mappings to SPDX
        mappings = {
            "mit license": "MIT",
            "mit": "MIT",
            "apache software license": "Apache-2.0",
            "apache license 2.0": "Apache-2.0",
            "apache 2.0": "Apache-2.0",
            "apache-2.0": "Apache-2.0",
            "bsd license": "BSD-3-Clause",
            "bsd": "BSD-3-Clause",
            "new bsd license": "BSD-3-Clause",
            "bsd 3-clause": "BSD-3-Clause",
            "bsd-3-clause": "BSD-3-Clause",
            "bsd 2-clause": "BSD-2-Clause",
            "bsd-2-clause": "BSD-2-Clause",
            "gnu general public license v2": "GPL-2.0",
            "gpl v2": "GPL-2.0",
            "gpl-2.0": "GPL-2.0",
            "gnu general public license v3": "GPL-3.0",
            "gpl v3": "GPL-3.0",
            "gpl-3.0": "GPL-3.0",
            "gnu lesser general public license v2.1": "LGPL-2.1",
            "lgpl-2.1": "LGPL-2.1",
            "gnu lesser general public license v3": "LGPL-3.0",
            "lgpl-3.0": "LGPL-3.0",
            "mozilla public license 2.0": "MPL-2.0",
            "mpl-2.0": "MPL-2.0",
            "isc license": "ISC",
            "isc": "ISC",
            "unlicense": "Unlicense",
            "public domain": "Unlicense",
        }

        normalized = license_str.lower().strip()
        return mappings.get(normalized, license_str)

    def check_compatibility(self, license1: str, license2: str) -> bool:
        """Check if two licenses are compatible."""
        if license1 == "Unknown" or license2 == "Unknown":
            return False

        if license1 not in COMPATIBLE_LICENSES:
            return False

        return license2 in COMPATIBLE_LICENSES[license1]["compatible_with"]

    def analyze_dependencies(self) -> Dict:
        """Analyze all dependencies for license compatibility."""
        packages = self.get_installed_packages()
        results = {
            "project_license": PROJECT_LICENSE,
            "total_dependencies": len(packages),
            "dependencies": {},
            "compatibility_issues": [],
            "risk_summary": {"low": 0, "medium": 0, "high": 0, "unknown": 0},
            "copyleft_dependencies": [],
            "unknown_licenses": [],
        }

        for package_name, version in packages.items():
            # Skip our own package
            if package_name.lower() in ["pulse-sdk", "pulse"]:
                continue

            license_str = self.get_package_license_from_pypi(package_name)
            normalized_license = self.normalize_license_name(license_str)

            # Check compatibility with our project license
            compatible = self.check_compatibility(PROJECT_LICENSE, normalized_license)

            dep_info = {
                "version": version,
                "license": normalized_license,
                "original_license_string": license_str,
                "compatible": compatible,
                "risk_level": "unknown",
            }

            # Determine risk level
            if normalized_license in COMPATIBLE_LICENSES:
                license_info = COMPATIBLE_LICENSES[normalized_license]
                dep_info["risk_level"] = license_info["risk_level"]
                dep_info["copyleft"] = license_info["copyleft"]
                dep_info["commercial_use"] = license_info["commercial_use"]

                # Track copyleft dependencies
                if license_info["copyleft"]:
                    results["copyleft_dependencies"].append(
                        {
                            "name": package_name,
                            "license": normalized_license,
                            "version": version,
                        }
                    )
            else:
                dep_info["risk_level"] = "unknown"
                results["unknown_licenses"].append(
                    {
                        "name": package_name,
                        "license": normalized_license,
                        "original": license_str,
                        "version": version,
                    }
                )

            # Track compatibility issues
            if not compatible:
                issue = {
                    "package": package_name,
                    "version": version,
                    "license": normalized_license,
                    "reason": (
                        f"License {normalized_license} not compatible with "
                        f"{PROJECT_LICENSE}"
                    ),
                }
                results["compatibility_issues"].append(issue)

            # Update risk summary
            results["risk_summary"][dep_info["risk_level"]] += 1
            results["dependencies"][package_name] = dep_info

        return results

    def generate_spdx_document(self, analysis_results: Dict) -> Dict:
        """Generate SPDX document for the project."""
        spdx_doc = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": "Pulse SDK",
            "documentNamespace": (
                f"https://github.com/researchwise/pulse-sdk/spdx-"
                f"{analysis_results.get('timestamp', 'unknown')}"
            ),
            "creationInfo": {
                "created": "2024-12-19T00:00:00Z",
                "creators": ["Tool: pulse-sdk-license-checker"],
                "licenseListVersion": "3.21",
            },
            "packages": [],
        }

        # Add main package
        main_package = {
            "SPDXID": "SPDXRef-Package-PulseSDK",
            "name": "pulse-sdk",
            "downloadLocation": "https://github.com/researchwise/pulse-sdk",
            "filesAnalyzed": False,
            "licenseConcluded": PROJECT_LICENSE,
            "licenseDeclared": PROJECT_LICENSE,
            "copyrightText": "Copyright (c) 2024 Researchwise AI",
        }
        spdx_doc["packages"].append(main_package)

        # Add dependencies
        for package_name, dep_info in analysis_results["dependencies"].items():
            dep_package = {
                "SPDXID": (
                    f"SPDXRef-Package-{package_name.replace('-', '').replace('_', '')}"
                ),
                "name": package_name,
                "versionInfo": dep_info["version"],
                "downloadLocation": f"https://pypi.org/project/{package_name}/",
                "filesAnalyzed": False,
                "licenseConcluded": dep_info["license"],
                "licenseDeclared": dep_info["license"],
                "copyrightText": "NOASSERTION",
            }
            spdx_doc["packages"].append(dep_package)

        return spdx_doc

    def generate_report(self, output_format: str = "json") -> str:
        """Generate comprehensive license report."""
        analysis = self.analyze_dependencies()
        analysis["timestamp"] = "2024-12-19T00:00:00Z"

        if output_format == "json":
            return json.dumps(analysis, indent=2)
        elif output_format == "spdx":
            spdx_doc = self.generate_spdx_document(analysis)
            return json.dumps(spdx_doc, indent=2)
        elif output_format == "text":
            return self._format_text_report(analysis)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _format_text_report(self, analysis: Dict) -> str:
        """Format analysis results as human-readable text."""
        report = []
        report.append("=" * 60)
        report.append("PULSE SDK LICENSE COMPATIBILITY REPORT")
        report.append("=" * 60)
        report.append(f"Project License: {analysis['project_license']}")
        report.append(f"Total Dependencies: {analysis['total_dependencies']}")
        report.append("")

        # Risk Summary
        report.append("RISK SUMMARY:")
        report.append("-" * 20)
        for risk_level, count in analysis["risk_summary"].items():
            report.append(f"{risk_level.upper()}: {count} packages")
        report.append("")

        # Compatibility Issues
        if analysis["compatibility_issues"]:
            report.append("COMPATIBILITY ISSUES:")
            report.append("-" * 25)
            for issue in analysis["compatibility_issues"]:
                report.append(f"❌ {issue['package']} ({issue['version']})")
                report.append(f"   License: {issue['license']}")
                report.append(f"   Reason: {issue['reason']}")
                report.append("")
        else:
            report.append("✅ No compatibility issues found!")
            report.append("")

        # Copyleft Dependencies
        if analysis["copyleft_dependencies"]:
            report.append("COPYLEFT DEPENDENCIES:")
            report.append("-" * 25)
            for dep in analysis["copyleft_dependencies"]:
                name, version, license = dep["name"], dep["version"], dep["license"]
                report.append(f"⚠️  {name} ({version}) - {license}")
            report.append("")

        # Unknown Licenses
        if analysis["unknown_licenses"]:
            report.append("UNKNOWN LICENSES:")
            report.append("-" * 20)
            for dep in analysis["unknown_licenses"]:
                report.append(f"❓ {dep['name']} ({dep['version']})")
                report.append(f"   License: {dep['license']}")
                report.append(f"   Original: {dep['original']}")
                report.append("")

        return "\n".join(report)


def main():
    """Main entry point for license checker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check license compatibility for Pulse SDK"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text", "spdx"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with error code if compatibility issues found",
    )

    args = parser.parse_args()

    checker = LicenseChecker()

    try:
        report = checker.generate_report(args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)

        # Check for issues if requested
        if args.fail_on_issues:
            analysis = checker.analyze_dependencies()
            if analysis["compatibility_issues"] or analysis["unknown_licenses"]:
                print("\n❌ License compatibility issues found!")
                sys.exit(1)
            else:
                print("\n✅ All licenses are compatible!")

    except Exception as e:
        print(f"Error generating license report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
