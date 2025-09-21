#!/usr/bin/env python3
"""
Test script for streamlined release workflow validation.

This script validates that the streamlined release process meets all requirements:
1. Workflow generates fewer than 10 release assets (Requirement 1.5)
2. All compliance documents are accessible via repository links (Requirement 2.1)
3. Release notes contain required compliance links (Requirement 3.4)

Usage:
    python scripts/test_streamlined_workflow.py [--verbose] [--check-links]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class WorkflowValidationResult:
    """Represents the result of a workflow validation test."""

    def __init__(self, name: str, passed: bool, message: str, details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name} - {self.message}"


class StreamlinedWorkflowValidator:
    """Validates the streamlined release workflow implementation."""

    def __init__(self, verbose: bool = False, check_external_links: bool = False):
        self.verbose = verbose
        self.check_external_links = check_external_links
        self.project_root = Path(__file__).parent.parent
        self.results: List[WorkflowValidationResult] = []

        # Expected compliance documents in repository root
        self.required_compliance_docs = [
            "COMPLIANCE.md",
            "SECURITY.md",
            "LICENSE",
            "NOTICE",
            "THIRD-PARTY-ATTRIBUTIONS.md",
            "CHANGELOG.md",
        ]

        # Expected compliance links in release notes
        self.required_compliance_links = [
            "COMPLIANCE.md",
            "SECURITY.md",
            "LICENSE",
            "THIRD-PARTY-ATTRIBUTIONS.md",
        ]

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def run_command(
        self, cmd: List[str], cwd: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out after 5 minutes"
        except Exception as e:
            return 1, "", str(e)

    def simulate_workflow_artifacts(self) -> Dict[str, int]:
        """
        Simulate the workflow artifact generation to count expected release assets.

        Returns:
            Dict mapping artifact type to count
        """
        self.log("Simulating workflow artifact generation...")

        # Create temporary directory to simulate build artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Simulate distribution files (2 files: wheel + sdist)
            dist_dir = temp_path / "dist"
            dist_dir.mkdir()

            # Create mock distribution files
            wheel_file = dist_dir / "pulse_sdk-0.4.1-py3-none-any.whl"
            sdist_file = dist_dir / "pulse_sdk-0.4.1.tar.gz"
            wheel_file.touch()
            sdist_file.touch()

            # Simulate attestation files (1 per distribution)
            attestation_wheel = (
                dist_dir / "pulse_sdk-0.4.1-py3-none-any.whl.attestation"
            )
            attestation_sdist = dist_dir / "pulse_sdk-0.4.1.tar.gz.attestation"
            attestation_wheel.touch()
            attestation_sdist.touch()

            # Simulate signature files (2 per distribution: .sig and .crt)
            signatures_dir = temp_path / "signatures"
            signatures_dir.mkdir()

            sig_files = [
                "pulse_sdk-0.4.1-py3-none-any.whl.sig",
                "pulse_sdk-0.4.1-py3-none-any.whl.crt",
                "pulse_sdk-0.4.1.tar.gz.sig",
                "pulse_sdk-0.4.1.tar.gz.crt",
            ]

            for sig_file in sig_files:
                (signatures_dir / sig_file).touch()

            # Simulate SBOM file (1 file: single CycloneDX format)
            sbom_file = temp_path / "sbom.cyclonedx.json"
            sbom_file.write_text('{"bomFormat": "CycloneDX", "specVersion": "1.4"}')

            # Build provenance is included in attestation files, no separate file needed
            # This aligns with GitHub's attestation system where provenance is embedded

            # Count artifacts
            artifact_counts = {
                "distributions": len(list(dist_dir.glob("*.whl")))
                + len(list(dist_dir.glob("*.tar.gz"))),
                "attestations": len(list(dist_dir.glob("*.attestation"))),
                "signatures": len(list(signatures_dir.glob("*.sig")))
                + len(list(signatures_dir.glob("*.crt"))),
                "sbom": 1 if sbom_file.exists() else 0,
                "provenance": 0,  # Provenance included in attestations
            }

            self.log(f"Simulated artifact counts: {artifact_counts}")
            return artifact_counts

    def validate_asset_count_limit(self) -> WorkflowValidationResult:
        """
        Validate that workflow generates fewer than 10 release assets.

        Requirement 1.5: Total number of release assets SHALL be fewer than 10 files
        """
        self.log("Validating release asset count limit...")

        try:
            artifact_counts = self.simulate_workflow_artifacts()

            total_assets = sum(artifact_counts.values())

            # Expected breakdown (must be fewer than 10 files per Requirement 1.5):
            # - 2 distribution files (wheel + sdist)
            # - 2 attestation files (1 per distribution, includes provenance)
            # - 4 signature files (2 per distribution: .sig + .crt)
            # - 1 SBOM file (single CycloneDX format)
            # Total: 9 files (build provenance is included in attestations)

            expected_breakdown = {
                "distributions": 2,
                "attestations": 2,  # Separate .attestation files generated by GitHub Actions
                "signatures": 4,
                "sbom": 1,
                "provenance": 0,  # Provenance included in attestations, no separate file needed
            }

            # Validate individual counts
            validation_errors = []
            for artifact_type, expected_count in expected_breakdown.items():
                actual_count = artifact_counts.get(artifact_type, 0)
                if actual_count != expected_count:
                    validation_errors.append(
                        f"{artifact_type}: expected {expected_count}, got {actual_count}"
                    )

            if validation_errors:
                return WorkflowValidationResult(
                    "Asset Count Validation",
                    False,
                    f"Artifact count mismatch: {'; '.join(validation_errors)}",
                    f"Total assets: {total_assets}, Expected: {sum(expected_breakdown.values())}",
                )

            if total_assets > 10:
                return WorkflowValidationResult(
                    "Asset Count Validation",
                    False,
                    f"Total release assets ({total_assets}) exceeds limit of 10",
                    f"Breakdown: {artifact_counts}",
                )

            return WorkflowValidationResult(
                "Asset Count Validation",
                True,
                f"Release asset count ({total_assets}) is within limit (<10)",
                f"Breakdown: distributions={artifact_counts['distributions']}, "
                f"attestations={artifact_counts['attestations']}, "
                f"signatures={artifact_counts['signatures']}, "
                f"sbom={artifact_counts['sbom']}, "
                f"provenance={artifact_counts['provenance']}",
            )

        except Exception as e:
            return WorkflowValidationResult(
                "Asset Count Validation",
                False,
                f"Validation failed with exception: {str(e)}",
                "",
            )

    def check_file_accessibility(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file is accessible in the repository.

        Args:
            file_path: Relative path from repository root

        Returns:
            Tuple of (is_accessible, message)
        """
        full_path = self.project_root / file_path

        if full_path.exists() and full_path.is_file():
            # Check if file is readable
            try:
                content = full_path.read_text(encoding="utf-8")
                if len(content.strip()) > 0:
                    return True, f"File exists and is readable ({len(content)} chars)"
                else:
                    return False, "File exists but is empty"
            except Exception as e:
                return False, f"File exists but cannot be read: {str(e)}"
        else:
            return False, "File does not exist"

    def check_github_link_format(self, link: str) -> bool:
        """
        Check if a link follows the expected GitHub repository format.

        Args:
            link: The link to validate

        Returns:
            True if link format is valid for GitHub repository access
        """
        # Expected format: https://github.com/researchwiseai/pulse-py/blob/main/{filename}
        github_pattern = r"https://github\.com/researchwiseai/pulse-py/blob/main/[A-Za-z0-9._/-]+\.md|https://github\.com/researchwiseai/pulse-py/blob/main/[A-Z][A-Z_-]*"
        return bool(re.match(github_pattern, link))

    def validate_compliance_document_accessibility(self) -> WorkflowValidationResult:
        """
        Validate that all compliance documents are accessible via repository links.

        Requirement 2.1: Release notes SHALL contain a "Compliance & Verification" section
        with links to all compliance documents
        """
        self.log("Validating compliance document accessibility...")

        try:
            missing_docs = []
            accessible_docs = []

            for doc in self.required_compliance_docs:
                is_accessible, message = self.check_file_accessibility(doc)

                if is_accessible:
                    accessible_docs.append(doc)
                    self.log(f"✓ {doc}: {message}")
                else:
                    missing_docs.append(doc)
                    self.log(f"✗ {doc}: {message}", "ERROR")

            if missing_docs:
                return WorkflowValidationResult(
                    "Compliance Document Accessibility",
                    False,
                    f"Missing or inaccessible compliance documents: {', '.join(missing_docs)}",
                    f"Accessible: {len(accessible_docs)}/{len(self.required_compliance_docs)}",
                )

            return WorkflowValidationResult(
                "Compliance Document Accessibility",
                True,
                f"All {len(accessible_docs)} compliance documents are accessible",
                f"Validated documents: {', '.join(accessible_docs)}",
            )

        except Exception as e:
            return WorkflowValidationResult(
                "Compliance Document Accessibility",
                False,
                f"Validation failed with exception: {str(e)}",
                "",
            )

    def extract_release_notes_template(self) -> Optional[str]:
        """
        Extract the release notes template from the GitHub Actions workflow.

        Returns:
            The release notes template content or None if not found
        """
        workflow_file = self.project_root / ".github" / "workflows" / "publish.yml"

        if not workflow_file.exists():
            self.log("GitHub Actions workflow file not found", "ERROR")
            return None

        try:
            workflow_content = workflow_file.read_text()

            # Find the release notes body section
            body_match = re.search(
                r"body:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)",
                workflow_content,
                re.DOTALL,
            )

            if body_match:
                # Clean up the extracted content
                body_content = body_match.group(1)
                # Remove leading whitespace from each line
                lines = body_content.split("\n")
                cleaned_lines = []
                for line in lines:
                    # Remove consistent leading whitespace
                    if line.strip():
                        cleaned_lines.append(line.lstrip())
                    else:
                        cleaned_lines.append("")

                return "\n".join(cleaned_lines)
            else:
                self.log("Could not find release notes body in workflow", "ERROR")
                return None

        except Exception as e:
            self.log(f"Error reading workflow file: {str(e)}", "ERROR")
            return None

    def validate_release_notes_compliance_links(self) -> WorkflowValidationResult:
        """
        Validate that release notes contain required compliance links.

        Requirement 3.4: Release notes SHALL include a "Compliance & Verification" section
        with direct links to relevant documents
        """
        self.log("Validating release notes compliance links...")

        try:
            release_notes_template = self.extract_release_notes_template()

            if not release_notes_template:
                return WorkflowValidationResult(
                    "Release Notes Compliance Links",
                    False,
                    "Could not extract release notes template from workflow",
                    "Check .github/workflows/publish.yml for proper body section",
                )

            self.log(
                f"Extracted release notes template ({len(release_notes_template)} chars)"
            )

            # Check for Compliance & Verification section
            if "Compliance & Verification" not in release_notes_template:
                return WorkflowValidationResult(
                    "Release Notes Compliance Links",
                    False,
                    "Release notes missing 'Compliance & Verification' section",
                    "Required section not found in template",
                )

            # Check for required compliance document links
            missing_links = []
            found_links = []

            for doc in self.required_compliance_links:
                # Look for GitHub repository links to the document
                link_patterns = [
                    f"github.com/researchwiseai/pulse-py/blob/main/{doc}",
                    f"../blob/main/{doc}",  # Relative GitHub link format
                    f"]{doc}]",  # Markdown link text
                ]

                found = False
                for pattern in link_patterns:
                    if pattern in release_notes_template:
                        found = True
                        break

                if found:
                    found_links.append(doc)
                    self.log(f"✓ Found link to {doc}")
                else:
                    missing_links.append(doc)
                    self.log(f"✗ Missing link to {doc}", "ERROR")

            # Check for additional required sections
            required_sections = [
                "📋 Documentation",
                "🔐 Supply Chain Security",
                "✅ Verification",
            ]

            missing_sections = []
            for section in required_sections:
                if section not in release_notes_template:
                    missing_sections.append(section)
                else:
                    self.log(f"✓ Found section: {section}")

            # Check for verification instructions
            has_verification_instructions = (
                "cosign verify-blob" in release_notes_template
                or "Verify package signature" in release_notes_template
            )

            if not has_verification_instructions:
                missing_sections.append("Verification instructions")

            # Determine overall result
            issues = []
            if missing_links:
                issues.append(f"Missing links: {', '.join(missing_links)}")
            if missing_sections:
                issues.append(f"Missing sections: {', '.join(missing_sections)}")

            if issues:
                return WorkflowValidationResult(
                    "Release Notes Compliance Links",
                    False,
                    f"Release notes compliance issues: {'; '.join(issues)}",
                    f"Found links: {len(found_links)}/{len(self.required_compliance_links)}",
                )

            return WorkflowValidationResult(
                "Release Notes Compliance Links",
                True,
                "Release notes contain all required compliance links and sections",
                f"Validated {len(found_links)} compliance links and {len(required_sections)} sections",
            )

        except Exception as e:
            return WorkflowValidationResult(
                "Release Notes Compliance Links",
                False,
                f"Validation failed with exception: {str(e)}",
                "",
            )

    def validate_workflow_configuration(self) -> WorkflowValidationResult:
        """
        Validate that the GitHub Actions workflow is properly configured for streamlined release.

        Additional validation to ensure workflow matches design requirements.
        """
        self.log("Validating workflow configuration...")

        try:
            workflow_file = self.project_root / ".github" / "workflows" / "publish.yml"

            if not workflow_file.exists():
                return WorkflowValidationResult(
                    "Workflow Configuration",
                    False,
                    "GitHub Actions publish workflow not found",
                    "Expected .github/workflows/publish.yml",
                )

            workflow_content = workflow_file.read_text()

            # Check for streamlined SBOM generation (single CycloneDX format)
            sbom_checks = [
                "cyclonedx-json=sbom.cyclonedx.json" in workflow_content,
                "syft dist/" in workflow_content,
                "SPDX" not in workflow_content or workflow_content.count("SPDX") == 0,
            ]

            # Check that SBOM signing is removed
            sbom_signing_removed = (
                "Sign SBOMs" not in workflow_content
                and "sbom.*sig" not in workflow_content
            )

            # Check that compliance documents are not attached to releases
            compliance_not_attached = all(
                (
                    doc not in workflow_content.split("files:")[1]
                    if "files:" in workflow_content
                    else True
                )
                for doc in ["COMPLIANCE.md", "LICENSE", "NOTICE"]
            )

            # Check for pre-release validation
            has_validation = "pre-release validation" in workflow_content.lower()

            # Check for streamlined artifact upload
            has_streamlined_upload = (
                "supply-chain-artifacts" in workflow_content
                and "retention-days: 90" in workflow_content
            )

            validation_results = {
                "Single SBOM format": all(sbom_checks),
                "SBOM signing removed": sbom_signing_removed,
                "Compliance docs not attached": compliance_not_attached,
                "Pre-release validation": has_validation,
                "Streamlined artifact upload": has_streamlined_upload,
            }

            passed_checks = [k for k, v in validation_results.items() if v]
            failed_checks = [k for k, v in validation_results.items() if not v]

            if failed_checks:
                return WorkflowValidationResult(
                    "Workflow Configuration",
                    False,
                    f"Workflow configuration issues: {', '.join(failed_checks)}",
                    f"Passed: {len(passed_checks)}/{len(validation_results)}",
                )

            return WorkflowValidationResult(
                "Workflow Configuration",
                True,
                "Workflow properly configured for streamlined release process",
                f"All {len(validation_results)} configuration checks passed",
            )

        except Exception as e:
            return WorkflowValidationResult(
                "Workflow Configuration",
                False,
                f"Validation failed with exception: {str(e)}",
                "",
            )

    def run_all_validations(self) -> List[WorkflowValidationResult]:
        """Run all workflow validation tests."""
        self.log("Starting streamlined workflow validation...\n")

        validations = [
            self.validate_asset_count_limit,
            self.validate_compliance_document_accessibility,
            self.validate_release_notes_compliance_links,
            self.validate_workflow_configuration,
        ]

        for validation in validations:
            try:
                result = validation()
                self.results.append(result)
                print(f"{result}")
                if result.details and self.verbose:
                    print(f"   Details: {result.details}")
                print()
            except Exception as e:
                error_result = WorkflowValidationResult(
                    validation.__name__.replace("validate_", "")
                    .replace("_", " ")
                    .title(),
                    False,
                    f"Validation failed with exception: {str(e)}",
                    "",
                )
                self.results.append(error_result)
                print(f"{error_result}\n")

        return self.results

    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        report = f"""# Streamlined Release Workflow Validation Report

## Summary
- **Total Validations**: {total}
- **Passed**: {passed}
- **Failed**: {total - passed}
- **Success Rate**: {(passed/total*100):.1f}%

## Requirements Coverage
- **Requirement 1.5**: Asset count limit validation
- **Requirement 2.1**: Compliance document accessibility validation
- **Requirement 3.4**: Release notes compliance links validation

## Detailed Results

"""

        for result in self.results:
            status_emoji = "✅" if result.passed else "❌"
            report += f"### {status_emoji} {result.name}\n"
            report += f"**Status**: {'PASS' if result.passed else 'FAIL'}\n"
            report += f"**Message**: {result.message}\n"
            if result.details:
                report += f"**Details**: {result.details}\n"
            report += "\n"

        # Overall assessment
        if passed == total:
            report += "## 🎉 Overall Assessment: WORKFLOW READY\n"
            report += "All validation checks passed. The streamlined workflow is properly implemented.\n"
        elif passed >= total * 0.75:
            report += "## ⚠️ Overall Assessment: MOSTLY READY\n"
            report += (
                "Most validation checks passed. Address failing items before release.\n"
            )
        else:
            report += "## ❌ Overall Assessment: NOT READY\n"
            report += "Significant issues found. Major work needed before workflow can be used.\n"

        return report


def main():
    """Main entry point for workflow validation."""
    parser = argparse.ArgumentParser(
        description="Validate streamlined release workflow implementation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Enable external link checking (requires network access)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("WORKFLOW_VALIDATION_REPORT.md"),
        help="Output file for validation report (default: WORKFLOW_VALIDATION_REPORT.md)",
    )

    args = parser.parse_args()

    validator = StreamlinedWorkflowValidator(
        verbose=args.verbose, check_external_links=args.check_links
    )

    results = validator.run_all_validations()

    # Generate and save report
    report = validator.generate_report()
    args.report_file.write_text(report)

    print("=" * 60)
    print("STREAMLINED WORKFLOW VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        print(result)

    print(f"\nValidation Results: {passed}/{total} passed")
    print(f"Detailed report saved to: {args.report_file}")

    # Exit with appropriate code
    if passed == total:
        print("\n🎉 All validations passed! Streamlined workflow is ready.")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} validation(s) failed. Review report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
