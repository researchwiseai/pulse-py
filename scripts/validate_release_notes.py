#!/usr/bin/env python3
"""
Automated verification script for release notes compliance.

This script validates that release notes contain all required compliance links
and sections as specified in the streamlined release process requirements.

Usage:
    python scripts/validate_release_notes.py [--workflow-file PATH] [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_release_notes_from_workflow(workflow_file: Path) -> str:
    """
    Extract the release notes template from GitHub Actions workflow.

    Args:
        workflow_file: Path to the GitHub Actions workflow file

    Returns:
        The release notes template content

    Raises:
        ValueError: If release notes template cannot be extracted
    """
    if not workflow_file.exists():
        raise ValueError(f"Workflow file not found: {workflow_file}")

    workflow_content = workflow_file.read_text()

    # Find the release notes body section
    body_match = re.search(
        r"body:\s*\|\s*\n(.*?)(?=\n\s*[a-zA-Z_-]+:|$)", workflow_content, re.DOTALL
    )

    if not body_match:
        raise ValueError("Could not find release notes body in workflow file")

    # Clean up the extracted content
    body_content = body_match.group(1)
    lines = body_content.split("\n")
    cleaned_lines = []

    for line in lines:
        # Remove consistent leading whitespace
        if line.strip():
            cleaned_lines.append(line.lstrip())
        else:
            cleaned_lines.append("")

    return "\n".join(cleaned_lines)


def validate_compliance_section(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes contain the required Compliance & Verification section.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for main Compliance & Verification section
    if "Compliance & Verification" not in release_notes:
        issues.append("Missing 'Compliance & Verification' section")
    elif verbose:
        print("✓ Found 'Compliance & Verification' section")

    # Check for required subsections
    required_subsections = [
        "📋 Documentation",
        "🔐 Supply Chain Security",
        "✅ Verification",
    ]

    for subsection in required_subsections:
        if subsection not in release_notes:
            issues.append(f"Missing subsection: {subsection}")
        elif verbose:
            print(f"✓ Found subsection: {subsection}")

    return len(issues) == 0, issues


def validate_compliance_links(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes contain links to all required compliance documents.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Required compliance documents that must be linked
    required_docs = [
        "COMPLIANCE.md",
        "SECURITY.md",
        "LICENSE",
        "THIRD-PARTY-ATTRIBUTIONS.md",
    ]

    for doc in required_docs:
        # Look for GitHub repository links to the document
        link_patterns = [
            f"github.com/researchwiseai/pulse-py/blob/main/{doc}",
            f"../blob/main/{doc}",  # Relative GitHub link format
            f"]{doc}]",  # Markdown link text
        ]

        found = any(pattern in release_notes for pattern in link_patterns)

        if not found:
            issues.append(f"Missing link to compliance document: {doc}")
        elif verbose:
            print(f"✓ Found link to {doc}")

    return len(issues) == 0, issues


def validate_verification_instructions(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes contain verification instructions.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for cosign verification instructions
    has_cosign_instructions = "cosign verify-blob" in release_notes
    has_verification_section = (
        "Verify package signature" in release_notes or "Verification" in release_notes
    )

    if not (has_cosign_instructions or has_verification_section):
        issues.append(
            "Missing verification instructions (should include cosign verify-blob command)"
        )
    elif verbose:
        print("✓ Found verification instructions")

    # Check for specific verification elements
    verification_elements = [
        "--certificate",
        "--signature",
        "--certificate-identity-regexp",
        "--certificate-oidc-issuer",
    ]

    missing_elements = []
    for element in verification_elements:
        if element not in release_notes:
            missing_elements.append(element)
        elif verbose:
            print(f"✓ Found verification element: {element}")

    if missing_elements:
        issues.append(
            f"Verification instructions missing elements: {', '.join(missing_elements)}"
        )

    return len(issues) == 0, issues


def validate_installation_instructions(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes contain installation instructions.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for Installation section
    if "## Installation" not in release_notes:
        issues.append("Missing '## Installation' section")
    elif verbose:
        print("✓ Found Installation section")

    # Check for pip install command
    if "pip install pulse-sdk" not in release_notes:
        issues.append("Missing pip install command")
    elif verbose:
        print("✓ Found pip install command")

    return len(issues) == 0, issues


def validate_changelog_reference(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes reference the changelog.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for changelog reference
    changelog_patterns = ["CHANGELOG.md", "changelog", "detailed change history"]

    found_changelog_ref = any(
        pattern.lower() in release_notes.lower() for pattern in changelog_patterns
    )

    if not found_changelog_ref:
        issues.append("Missing reference to CHANGELOG.md or change history")
    elif verbose:
        print("✓ Found changelog reference")

    return len(issues) == 0, issues


def validate_supply_chain_info(
    release_notes: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that release notes contain supply chain security information.

    Args:
        release_notes: The release notes content
        verbose: Whether to print detailed validation info

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for SBOM reference
    if "SBOM" not in release_notes or "sbom.cyclonedx.json" not in release_notes:
        issues.append("Missing SBOM reference (should mention sbom.cyclonedx.json)")
    elif verbose:
        print("✓ Found SBOM reference")

    # Check for build provenance reference (can be in attestations or separate file)
    has_provenance_ref = (
        "build-provenance.json" in release_notes
        or "Build Provenance" in release_notes
        or "provenance" in release_notes.lower()
    )

    if not has_provenance_ref:
        issues.append("Missing build provenance reference")
    elif verbose:
        print("✓ Found build provenance reference")

    # Check for signature reference
    if "Signatures" not in release_notes and "signature" not in release_notes.lower():
        issues.append("Missing signature reference")
    elif verbose:
        print("✓ Found signature reference")

    return len(issues) == 0, issues


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate release notes compliance with streamlined workflow requirements"
    )
    parser.add_argument(
        "--workflow-file",
        type=Path,
        default=Path(".github/workflows/publish.yml"),
        help="Path to GitHub Actions workflow file (default: .github/workflows/publish.yml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Extract release notes from workflow
        if args.verbose:
            print(f"Extracting release notes from: {args.workflow_file}")

        release_notes = extract_release_notes_from_workflow(args.workflow_file)

        if args.verbose:
            print(f"Extracted release notes template ({len(release_notes)} characters)")
            print("-" * 60)

        # Run all validations
        validations = [
            ("Compliance Section", validate_compliance_section),
            ("Compliance Links", validate_compliance_links),
            ("Verification Instructions", validate_verification_instructions),
            ("Installation Instructions", validate_installation_instructions),
            ("Changelog Reference", validate_changelog_reference),
            ("Supply Chain Info", validate_supply_chain_info),
        ]

        all_passed = True
        total_issues = []

        print("🔍 Validating release notes compliance...\n")

        for validation_name, validation_func in validations:
            if args.verbose:
                print(f"Running {validation_name} validation...")

            is_valid, issues = validation_func(release_notes, args.verbose)

            if is_valid:
                print(f"✅ {validation_name}: PASS")
            else:
                print(f"❌ {validation_name}: FAIL")
                for issue in issues:
                    print(f"   - {issue}")
                all_passed = False
                total_issues.extend(issues)

            if args.verbose:
                print()

        # Print summary
        print("\n" + "=" * 60)
        print("RELEASE NOTES VALIDATION SUMMARY")
        print("=" * 60)

        if all_passed:
            print("🎉 All validations passed! Release notes are compliant.")
            print("\nThe release notes template contains all required:")
            print("- Compliance & Verification section with proper subsections")
            print("- Links to all required compliance documents")
            print("- Verification instructions with cosign commands")
            print("- Installation instructions")
            print("- Changelog reference")
            print("- Supply chain security information")
            return 0
        else:
            print(f"❌ {len(total_issues)} validation issues found:")
            for i, issue in enumerate(total_issues, 1):
                print(f"  {i}. {issue}")

            print(
                "\nPlease update the release notes template in the GitHub Actions workflow"
            )
            print(
                "to address these issues before using the streamlined release process."
            )
            return 1

    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
