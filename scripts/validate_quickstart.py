#!/usr/bin/env python3
"""
Quickstart guide validator for Pulse SDK.

This script validates that the quickstart guide contains all necessary components
and that the steps can be followed successfully.

Usage:
    python scripts/validate_quickstart.py [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class QuickstartValidator:
    """Validates the quickstart guide for completeness and accuracy."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def add_error(self, message: str):
        """Add an error."""
        self.errors.append(message)
        self.log(message, "ERROR")

    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)
        self.log(message, "WARNING")

    def extract_code_blocks(self, content: str) -> Dict[str, List[Tuple[str, int]]]:
        """Extract code blocks by language."""
        code_blocks = {"python": [], "bash": [], "shell": []}
        lines = content.split("\n")
        in_code_block = False
        current_block = []
        block_start_line = 0
        language = None

        for i, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    block_start_line = i
                    # Extract language if specified
                    lang_match = re.match(r"```(\w+)", line.strip())
                    language = lang_match.group(1) if lang_match else "text"
                    current_block = []
                else:
                    # Ending a code block
                    in_code_block = False
                    if language in code_blocks and current_block:
                        code = "\n".join(current_block)
                        code_blocks[language].append((code, block_start_line))
                    current_block = []
                    language = None
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def validate_installation_commands(
        self, bash_blocks: List[Tuple[str, int]]
    ) -> bool:
        """Validate installation commands in the quickstart guide."""
        self.log("Validating installation commands...")

        required_commands = [
            r"pip install pulse-sdk\[all\]",
            r"pip install pulse-sdk\[minimal\]",
            r"pip install pulse-sdk\[.*\]",  # Any variation
        ]

        found_commands = []
        for code, line_num in bash_blocks:
            for pattern in required_commands:
                if re.search(pattern, code):
                    found_commands.append(pattern)
                    self.log(
                        f"✓ Found installation command at line {line_num}: {pattern}"
                    )

        if not found_commands:
            self.add_error("No installation commands found in quickstart guide")
            return False

        # Check for different installation options
        expected_options = ["all", "minimal", "analysis", "dev"]
        found_options = []

        for code, line_num in bash_blocks:
            for option in expected_options:
                if f"pulse-sdk[{option}]" in code or f"pulse-sdk[.*{option}.*]" in code:
                    found_options.append(option)

        missing_options = set(expected_options) - set(found_options)
        if missing_options:
            self.add_warning(f"Missing installation options: {missing_options}")

        return True

    def validate_python_examples(self, python_blocks: List[Tuple[str, int]]) -> bool:
        """Validate Python code examples in the quickstart guide."""
        self.log("Validating Python examples...")

        # Required imports and patterns
        required_patterns = [
            (r"from pulse\.core\.client import CoreClient", "CoreClient import"),
            (
                r"from pulse\.starters import sentiment_analysis",
                "sentiment_analysis import",
            ),
            (r"from pulse\.starters import summarize", "summarize import"),
            (
                r"from pulse\.starters import cluster_analysis",
                "cluster_analysis import",
            ),
            (r"sentiment_analysis\(", "sentiment_analysis usage"),
            (r"summarize\(", "summarize usage"),
            (r"cluster_analysis\(", "cluster_analysis usage"),
        ]

        found_patterns = []
        for code, line_num in python_blocks:
            for pattern, description in required_patterns:
                if re.search(pattern, code):
                    found_patterns.append(description)
                    self.log(f"✓ Found {description} at line {line_num}")

        missing_patterns = [
            desc for pattern, desc in required_patterns if desc not in found_patterns
        ]

        if missing_patterns:
            for missing in missing_patterns:
                self.add_warning(f"Missing example: {missing}")

        # Check for authentication examples
        auth_patterns = [
            r"ClientCredentialsAuth",
            r"AuthorizationCodePKCEAuth",
            r"PULSE_CLIENT_ID",
            r"PULSE_CLIENT_SECRET",
        ]

        found_auth = False
        for code, line_num in python_blocks:
            for pattern in auth_patterns:
                if re.search(pattern, code):
                    found_auth = True
                    break

        if not found_auth:
            self.add_warning("No authentication examples found")

        return (
            len(missing_patterns) < len(required_patterns) // 2
        )  # Allow some flexibility

    def validate_structure_and_content(self, content: str) -> bool:
        """Validate the overall structure and content of the quickstart guide."""
        self.log("Validating quickstart structure and content...")

        # Required sections
        required_sections = [
            (r"#.*[Qq]uick [Ss]tart", "Quick Start title"),
            (r"#.*[Ii]nstall", "Installation section"),
            (r"#.*[Aa]uthentication", "Authentication section"),
            (r"#.*[Tt]roubleshooting", "Troubleshooting section"),
        ]

        found_sections = []
        for pattern, description in required_sections:
            if re.search(pattern, content, re.MULTILINE):
                found_sections.append(description)
                self.log(f"✓ Found {description}")
            else:
                self.add_warning(f"Missing section: {description}")

        # Check for time estimates
        time_patterns = [
            r"\d+\s*minutes?",
            r"\d+\s*seconds?",
            r"5-minute",
            r"under \d+ minutes?",
        ]

        found_time_estimate = False
        for pattern in time_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_time_estimate = True
                self.log("✓ Found time estimates")
                break

        if not found_time_estimate:
            self.add_warning("No time estimates found for setup steps")

        # Check for common use cases
        use_case_patterns = [
            r"[Cc]ustomer [Ff]eedback",
            r"[Ss]entiment [Aa]nalysis",
            r"[Tt]heme [Aa]nalysis",
            r"[Cc]lustering",
            r"[Ss]ummariz",
        ]

        found_use_cases = []
        for pattern in use_case_patterns:
            if re.search(pattern, content):
                found_use_cases.append(pattern)

        if len(found_use_cases) < 3:
            self.add_warning("Limited use case examples found")
        else:
            self.log(f"✓ Found {len(found_use_cases)} use case examples")

        return len(found_sections) >= len(required_sections) // 2

    def validate_links_and_references(self, content: str) -> bool:
        """Validate that the quickstart guide has proper links and references."""
        self.log("Validating links and references...")

        # Required links/references
        required_links = [
            (r"\[.*installation.*\]", "Installation guide link"),
            (r"\[.*authentication.*\]", "Authentication guide link"),
            (r"\[.*examples.*\]", "Examples link"),
            (r"\[.*documentation.*\]", "Documentation link"),
        ]

        found_links = []
        for pattern, description in required_links:
            if re.search(pattern, content, re.IGNORECASE):
                found_links.append(description)
                self.log(f"✓ Found {description}")
            else:
                self.add_warning(f"Missing link: {description}")

        # Check for GitHub references
        github_patterns = [
            r"github\.com/researchwiseai/pulse-py",
            r"github\.com.*pulse.*py",
            r"issues",
            r"examples",
        ]

        found_github = False
        for pattern in github_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_github = True
                break

        if found_github:
            self.log("✓ Found GitHub references")
        else:
            self.add_warning("No GitHub references found")

        return len(found_links) >= len(required_links) // 2

    def validate_error_handling_guidance(self, content: str) -> bool:
        """Validate that error handling and troubleshooting guidance is present."""
        self.log("Validating error handling guidance...")

        error_patterns = [
            r"ImportError",
            r"PulseAPIError",
            r"401 Unauthorized",
            r"timeout",
            r"connection",
            r"[Ee]rror.*[Ss]olution",
            r"[Pp]roblem.*[Ss]olution",
            r"[Tt]roubleshooting",
        ]

        found_errors = []
        for pattern in error_patterns:
            if re.search(pattern, content):
                found_errors.append(pattern)

        if len(found_errors) >= 3:
            self.log(f"✓ Found {len(found_errors)} error handling examples")
            return True
        else:
            self.add_warning("Limited error handling guidance found")
            return False

    def validate_quickstart_guide(self, file_path: Path) -> bool:
        """Validate the complete quickstart guide."""
        self.log(f"Validating quickstart guide: {file_path}")

        if not file_path.exists():
            self.add_error(f"Quickstart guide not found: {file_path}")
            return False

        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract code blocks
            code_blocks = self.extract_code_blocks(content)

            # Run all validations
            validations = [
                self.validate_installation_commands(code_blocks["bash"]),
                self.validate_python_examples(code_blocks["python"]),
                self.validate_structure_and_content(content),
                self.validate_links_and_references(content),
                self.validate_error_handling_guidance(content),
            ]

            success = all(validations)

            # Print summary
            self.log("\nQuickstart Validation Summary:")
            self.log(f"  Errors: {len(self.errors)}")
            self.log(f"  Warnings: {len(self.warnings)}")

            if self.errors:
                self.log("\nErrors:")
                for error in self.errors:
                    self.log(f"  - {error}")

            if self.warnings:
                self.log("\nWarnings:")
                for warning in self.warnings:
                    self.log(f"  - {warning}")

            return success

        except Exception as e:
            self.add_error(f"Error reading quickstart guide: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Pulse SDK quickstart guide")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("docs/quickstart.md"),
        help="Quickstart guide file (default: docs/quickstart.md)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    validator = QuickstartValidator(verbose=args.verbose)
    success = validator.validate_quickstart_guide(args.file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
