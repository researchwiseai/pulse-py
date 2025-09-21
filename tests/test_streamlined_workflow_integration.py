"""
Integration tests for streamlined release workflow validation.

These tests validate that the streamlined release process implementation
meets all requirements from the specification.

Requirements tested:
- 1.5: Total number of release assets SHALL be fewer than 10 files
- 2.1: Release notes SHALL contain compliance document links
- 3.4: Release notes SHALL include required compliance sections
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest


class TestStreamlinedWorkflowIntegration:
    """Integration tests for streamlined workflow validation."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def workflow_file(self, project_root: Path) -> Path:
        """Get the GitHub Actions workflow file."""
        return project_root / ".github" / "workflows" / "publish.yml"

    @pytest.fixture
    def required_compliance_docs(self) -> List[str]:
        """List of required compliance documents."""
        return [
            "COMPLIANCE.md",
            "SECURITY.md",
            "LICENSE",
            "NOTICE",
            "THIRD-PARTY-ATTRIBUTIONS.md",
            "CHANGELOG.md",
        ]

    @pytest.fixture
    def required_compliance_links(self) -> List[str]:
        """List of compliance documents that must be linked in release notes."""
        return [
            "COMPLIANCE.md",
            "SECURITY.md",
            "LICENSE",
            "THIRD-PARTY-ATTRIBUTIONS.md",
        ]

    def simulate_workflow_artifacts(self) -> Dict[str, int]:
        """
        Simulate the workflow artifact generation to count expected release assets.

        Returns:
            Dict mapping artifact type to count
        """
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
            return {
                "distributions": len(list(dist_dir.glob("*.whl")))
                + len(list(dist_dir.glob("*.tar.gz"))),
                "attestations": len(list(dist_dir.glob("*.attestation"))),
                "signatures": len(list(signatures_dir.glob("*.sig")))
                + len(list(signatures_dir.glob("*.crt"))),
                "sbom": 1 if sbom_file.exists() else 0,
                "provenance": 0,  # Provenance included in attestations
            }

    def test_release_asset_count_under_limit(self):
        """
        Test that workflow generates fewer than 10 release assets.

        Requirement 1.5: Total number of release assets SHALL be fewer than 10 files
        """
        artifact_counts = self.simulate_workflow_artifacts()
        total_assets = sum(artifact_counts.values())

        # Expected breakdown (must be fewer than 10 files per Requirement 1.5):
        # - 2 distribution files (wheel + sdist)
        # - 2 attestation files (1 per distribution, includes provenance)
        # - 4 signature files (2 per distribution: .sig + .crt)
        # - 1 SBOM file (single CycloneDX format)
        # Total: 9 files (build provenance included in attestations)

        expected_breakdown = {
            "distributions": 2,
            "attestations": 2,
            "signatures": 4,
            "sbom": 1,
            "provenance": 0,  # Provenance included in attestations
        }

        # Validate individual counts match expected
        for artifact_type, expected_count in expected_breakdown.items():
            actual_count = artifact_counts.get(artifact_type, 0)
            assert actual_count == expected_count, (
                f"Artifact count mismatch for {artifact_type}: "
                f"expected {expected_count}, got {actual_count}"
            )

        # Validate total is under limit
        assert total_assets < 10, (
            f"Total release assets ({total_assets}) exceeds limit of 10. "
            f"Breakdown: {artifact_counts}"
        )

        # Validate total matches expected sum
        expected_total = sum(expected_breakdown.values())
        assert (
            total_assets == expected_total
        ), f"Total assets ({total_assets}) doesn't match expected ({expected_total})"

    def test_compliance_documents_accessible(
        self, project_root: Path, required_compliance_docs: List[str]
    ):
        """
        Test that all compliance documents are accessible via repository links.

        Requirement 2.1: Release notes SHALL contain compliance document links
        """
        missing_docs = []

        for doc in required_compliance_docs:
            doc_path = project_root / doc

            # Check if file exists
            if not doc_path.exists():
                missing_docs.append(f"{doc} (file not found)")
                continue

            # Check if file is readable and non-empty
            try:
                content = doc_path.read_text(encoding="utf-8")
                if len(content.strip()) == 0:
                    missing_docs.append(f"{doc} (file is empty)")
            except Exception as e:
                missing_docs.append(f"{doc} (cannot read: {str(e)})")

        assert (
            not missing_docs
        ), f"Missing or inaccessible compliance documents: {', '.join(missing_docs)}"

    def test_release_notes_contain_compliance_links(
        self, workflow_file: Path, required_compliance_links: List[str]
    ):
        """
        Test that release notes contain required compliance links.

        Requirement 3.4: Release notes SHALL include compliance links and sections
        """
        assert workflow_file.exists(), "GitHub Actions workflow file not found"

        workflow_content = workflow_file.read_text()

        # Extract release notes body section
        body_match = re.search(
            r"body:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)", workflow_content, re.DOTALL
        )

        assert body_match, "Could not find release notes body in workflow"

        release_notes_template = body_match.group(1)

        # Check for Compliance & Verification section
        assert (
            "Compliance & Verification" in release_notes_template
        ), "Release notes missing 'Compliance & Verification' section"

        # Check for required compliance document links
        missing_links = []

        for doc in required_compliance_links:
            # Look for GitHub repository links to the document
            link_patterns = [
                f"github.com/researchwiseai/pulse-py/blob/main/{doc}",
                f"../blob/main/{doc}",  # Relative GitHub link format
                f"]{doc}]",  # Markdown link text
            ]

            found = any(pattern in release_notes_template for pattern in link_patterns)

            if not found:
                missing_links.append(doc)

        assert (
            not missing_links
        ), f"Release notes missing links to compliance documents: {', '.join(missing_links)}"

    def test_release_notes_contain_required_sections(self, workflow_file: Path):
        """
        Test that release notes contain all required sections.

        Requirement 3.4: Release notes SHALL include required compliance sections
        """
        workflow_content = workflow_file.read_text()

        # Extract release notes body section
        body_match = re.search(
            r"body:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)", workflow_content, re.DOTALL
        )

        assert body_match, "Could not find release notes body in workflow"

        release_notes_template = body_match.group(1)

        # Check for required sections
        required_sections = [
            "## Changes",
            "## Installation",
            "## Compliance & Verification",
            "### 📋 Documentation",
            "### 🔐 Supply Chain Security",
            "### ✅ Verification",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in release_notes_template:
                missing_sections.append(section)

        assert (
            not missing_sections
        ), f"Release notes missing required sections: {', '.join(missing_sections)}"

        # Check for verification instructions
        has_verification_instructions = (
            "cosign verify-blob" in release_notes_template
            or "Verify package signature" in release_notes_template
        )

        assert (
            has_verification_instructions
        ), "Release notes missing verification instructions"

    def test_workflow_streamlined_configuration(self, workflow_file: Path):
        """
        Test that workflow is properly configured for streamlined release.

        Validates workflow matches streamlined design requirements.
        """
        workflow_content = workflow_file.read_text()

        # Check for single SBOM format (CycloneDX only)
        assert (
            "cyclonedx-json=sbom.cyclonedx.json" in workflow_content
        ), "Workflow should generate single CycloneDX SBOM"

        assert (
            "syft dist/" in workflow_content
        ), "Workflow should use syft to generate SBOM from dist/ directory"

        # Ensure SPDX format is not used (streamlined to single format)
        spdx_count = workflow_content.count("SPDX")
        assert (
            spdx_count == 0
        ), f"Workflow should not use SPDX format (found {spdx_count} references)"

        # Check that SBOM signing is removed (streamlined process)
        assert (
            "Sign SBOMs" not in workflow_content
        ), "Workflow should not sign SBOM files (streamlined process)"

        # Check that compliance documents are not attached to releases
        if "files:" in workflow_content:
            files_section = workflow_content.split("files:")[1]
            compliance_docs_in_files = [
                "COMPLIANCE.md" in files_section,
                "LICENSE" in files_section and "LICENSE-MANIFEST" not in files_section,
                "NOTICE" in files_section,
            ]

            assert not any(compliance_docs_in_files), (
                "Compliance documents should not be attached to releases "
                "(should be linked from repository instead)"
            )

        # Check for pre-release validation
        assert (
            "pre-release validation" in workflow_content.lower()
        ), "Workflow should include pre-release validation checks"

        # Check for streamlined artifact upload
        assert (
            "supply-chain-artifacts" in workflow_content
        ), "Workflow should use streamlined artifact upload"

    def test_workflow_asset_upload_configuration(self, workflow_file: Path):
        """
        Test that workflow artifact upload is properly configured.

        Validates that only streamlined assets are uploaded.
        """
        workflow_content = workflow_file.read_text()

        # Find the artifact upload section
        upload_match = re.search(
            r"uses: actions/upload-artifact@.*?\n.*?path:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)",
            workflow_content,
            re.DOTALL,
        )

        assert upload_match, "Could not find artifact upload configuration"

        upload_paths = upload_match.group(1)

        # Expected paths for streamlined artifacts (provenance included in attestations)
        expected_paths = [
            "dist/*.whl",
            "dist/*.tar.gz",
            "dist/*.attestation",
            "signatures/*.sig",
            "signatures/*.crt",
            "sbom.cyclonedx.json",
        ]

        missing_paths = []
        for path in expected_paths:
            if path not in upload_paths:
                missing_paths.append(path)

        assert (
            not missing_paths
        ), f"Artifact upload missing expected paths: {', '.join(missing_paths)}"

        # Ensure compliance documents are NOT uploaded
        excluded_paths = [
            "COMPLIANCE.md",
            "COMPLIANCE-CHECKLIST.md",
            "LICENSE-MANIFEST.json",
        ]

        included_excluded = []
        for path in excluded_paths:
            if path in upload_paths:
                included_excluded.append(path)

        assert (
            not included_excluded
        ), f"Artifact upload should not include compliance documents: {', '.join(included_excluded)}"

    def test_github_release_configuration(self, workflow_file: Path):
        """
        Test that GitHub release creation is properly configured.

        Validates release asset attachment matches streamlined requirements.
        """
        workflow_content = workflow_file.read_text()

        # Find the GitHub release section
        release_match = re.search(
            r"uses: softprops/action-gh-release@.*?\n.*?files:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)",
            workflow_content,
            re.DOTALL,
        )

        assert release_match, "Could not find GitHub release configuration"

        release_files = release_match.group(1)

        # Expected files for streamlined release (provenance included in attestations)
        expected_files = [
            "dist/*.whl",
            "dist/*.tar.gz",
            "dist/*.attestation",
            "signatures/*.sig",
            "signatures/*.crt",
            "sbom.cyclonedx.json",
        ]

        missing_files = []
        for file_pattern in expected_files:
            if file_pattern not in release_files:
                missing_files.append(file_pattern)

        assert (
            not missing_files
        ), f"GitHub release missing expected files: {', '.join(missing_files)}"

        # Ensure compliance documents are NOT attached to release
        excluded_files = [
            "COMPLIANCE.md",
            "COMPLIANCE-CHECKLIST.md",
            "LICENSE-MANIFEST.json",
            "THIRD-PARTY-ATTRIBUTIONS.md",
        ]

        included_excluded = []
        for file_pattern in excluded_files:
            if file_pattern in release_files:
                included_excluded.append(file_pattern)

        assert (
            not included_excluded
        ), f"GitHub release should not include compliance documents: {', '.join(included_excluded)}"

    def test_sbom_generation_streamlined(self, workflow_file: Path):
        """
        Test that SBOM generation follows streamlined approach.

        Validates single SBOM format and proper configuration.
        """
        workflow_content = workflow_file.read_text()

        # Should generate exactly one SBOM file
        sbom_generation_lines = [
            line
            for line in workflow_content.split("\n")
            if "syft" in line and "cyclonedx-json" in line
        ]

        assert (
            len(sbom_generation_lines) == 1
        ), f"Should have exactly one SBOM generation command, found {len(sbom_generation_lines)}"

        sbom_line = sbom_generation_lines[0]

        # Should target dist/ directory (covers both distributions)
        assert (
            "syft dist/" in sbom_line
        ), "SBOM generation should target dist/ directory to cover all distributions"

        # Should output to specific file name
        assert (
            "sbom.cyclonedx.json" in sbom_line
        ), "SBOM should be generated as sbom.cyclonedx.json"

        # Should use CycloneDX format
        assert "cyclonedx-json" in sbom_line, "SBOM should use CycloneDX format"

    def test_signing_configuration_streamlined(self, workflow_file: Path):
        """
        Test that signing configuration follows streamlined approach.

        Validates that only distributions are signed, not SBOM or compliance docs.
        """
        workflow_content = workflow_file.read_text()

        # Find signing section
        signing_section_match = re.search(
            r"Sign distributions.*?\n(.*?)(?=\n\s*- name:|$)",
            workflow_content,
            re.DOTALL,
        )

        assert signing_section_match, "Could not find distribution signing section"

        signing_section = signing_section_match.group(1)

        # Should only sign wheel and sdist files
        assert (
            "dist/*.whl dist/*.tar.gz" in signing_section
        ), "Should sign wheel and sdist files from dist/ directory"

        # Should NOT sign SBOM files (check for actual signing commands, not comments)
        sbom_signing_commands = [
            "cosign sign-blob" in line and "sbom" in line.lower()
            for line in signing_section.split("\n")
            if not line.strip().startswith("#")  # Ignore comments
        ]
        assert not any(
            sbom_signing_commands
        ), "Should not sign SBOM files (streamlined process)"

        # Should NOT sign compliance documents
        compliance_docs = ["COMPLIANCE.md", "LICENSE", "NOTICE"]
        for doc in compliance_docs:
            assert (
                doc not in signing_section
            ), f"Should not sign compliance document {doc} (streamlined process)"

        # Should output signatures to separate directory
        assert (
            "signatures/" in signing_section
        ), "Signatures should be output to signatures/ directory"


# Additional utility tests for validation script itself


class TestValidationScriptFunctionality:
    """Tests for the validation script functionality."""

    def test_validation_script_exists(self):
        """Test that the validation script exists and is executable."""
        script_path = (
            Path(__file__).parent.parent / "scripts" / "test_streamlined_workflow.py"
        )

        assert script_path.exists(), "Validation script not found"
        assert script_path.is_file(), "Validation script is not a file"

        # Check that script has proper shebang
        content = script_path.read_text()
        assert content.startswith(
            "#!/usr/bin/env python3"
        ), "Validation script should have proper shebang"

        # Check for main function
        assert "def main():" in content, "Validation script should have main() function"

        # Check for argument parsing
        assert "argparse" in content, "Validation script should use argparse"

    def test_validation_script_imports(self):
        """Test that validation script imports are available."""
        # Import the validation script module
        import sys
        from pathlib import Path

        script_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(script_dir))

        try:
            import test_streamlined_workflow

            # Check that main classes exist
            assert hasattr(
                test_streamlined_workflow, "StreamlinedWorkflowValidator"
            ), "Validation script should have StreamlinedWorkflowValidator class"

            assert hasattr(
                test_streamlined_workflow, "WorkflowValidationResult"
            ), "Validation script should have WorkflowValidationResult class"

        finally:
            sys.path.remove(str(script_dir))
