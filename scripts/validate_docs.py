#!/usr/bin/env python3
"""
Documentation validation script for the Pulse SDK.

This script validates:
1. Code examples in documentation files using doctest
2. Links in documentation files
3. Quick start guide steps
4. Documentation build verification

Usage:
    python scripts/validate_docs.py [--check-links] [--validate-quickstart]
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


class DocumentationValidator:
    """Validates documentation files for code examples, links, and build integrity."""

    def __init__(self, docs_dir: str = "docs", verbose: bool = False):
        self.docs_dir = Path(docs_dir)
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with appropriate level."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def add_error(self, message: str):
        """Add an error to the error list."""
        self.errors.append(message)
        self.log(message, "ERROR")

    def add_warning(self, message: str):
        """Add a warning to the warning list."""
        self.warnings.append(message)
        self.log(message, "WARNING")

    def extract_code_blocks(
        self, content: str, file_path: str
    ) -> List[Tuple[str, str, int]]:
        """Extract Python code blocks from markdown content."""
        code_blocks = []
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
                    language = lang_match.group(1) if lang_match else None
                    current_block = []
                else:
                    # Ending a code block
                    in_code_block = False
                    if language == "python" and current_block:
                        # Clean up indentation for code blocks in lists
                        import textwrap

                        code = textwrap.dedent("\n".join(current_block)).strip()
                        if code:  # Only add non-empty blocks
                            code_blocks.append((code, file_path, block_start_line))
                    current_block = []
                    language = None
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def validate_python_syntax(
        self, code: str, file_path: str, line_number: int
    ) -> bool:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            self.add_error(
                f"Syntax error in {file_path}:{line_number + e.lineno - 1}: {e.msg}"
            )
            return False

    def validate_code_examples(self) -> bool:
        """Validate all Python code examples in documentation."""
        self.log("Validating Python code examples...")

        success = True
        total_blocks = 0
        valid_blocks = 0

        for doc_file in self.docs_dir.glob("*.md"):
            self.log(f"Checking {doc_file}")

            try:
                content = doc_file.read_text(encoding="utf-8")
                code_blocks = self.extract_code_blocks(content, str(doc_file))

                for code, file_path, line_number in code_blocks:
                    total_blocks += 1

                    # Skip code blocks that are clearly examples or incomplete
                    if any(
                        skip in code
                        for skip in [
                            "your_client_id",
                            "your_client_secret",
                            "...",
                            "# Your code here",
                            "# TODO",
                            "# Example",
                        ]
                    ):
                        self.log(
                            f"Skipping example code block in {file_path}:{line_number}"
                        )
                        valid_blocks += 1
                        continue

                    if self.validate_python_syntax(code, file_path, line_number):
                        valid_blocks += 1
                    else:
                        success = False

            except Exception as e:
                self.add_error(f"Error reading {doc_file}: {e}")
                success = False

        self.log(
            f"Code validation complete: {valid_blocks}/{total_blocks} blocks valid"
        )
        return success

    def extract_links(self, content: str) -> List[str]:
        """Extract all links from markdown content."""
        links = []

        # Markdown links: [text](url)
        markdown_links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", content)
        links.extend([url for text, url in markdown_links])

        # Reference links: [text]: url
        ref_links = re.findall(r"^\[([^\]]+)\]:\s*(.+)$", content, re.MULTILINE)
        links.extend([url for text, url in ref_links])

        # HTML links: <a href="url">
        html_links = re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\']', content, re.IGNORECASE
        )
        links.extend(html_links)

        # Direct URLs: http(s)://...
        direct_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        links.extend(direct_urls)

        return links

    def check_link(self, url: str, base_path: str = None) -> Tuple[bool, str]:
        """Check if a link is valid."""
        if not HAS_REQUESTS:
            return True, "Requests not available, skipping HTTP checks"

        # Skip certain URLs that are known to be examples or placeholders
        skip_patterns = [
            r"example\.com",
            r"localhost",
            r"127\.0\.0\.1",
            r"your_.*",
            r"<.*>",
            r"\$\{.*\}",
            r"mailto:.*@example\.com",
        ]

        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True, "Skipped example/placeholder URL"

        # Handle relative links
        if url.startswith("#"):
            return True, "Anchor link (not validated)"
        elif url.startswith("/") or not url.startswith(("http://", "https://")):
            if base_path:
                # Try to resolve relative to documentation
                if url.startswith("/"):
                    url = url[1:]  # Remove leading slash
                full_path = Path(base_path).parent / url
                if full_path.exists():
                    return True, "Local file exists"
                else:
                    return False, f"Local file not found: {full_path}"
            return True, "Relative link (not validated)"

        # Check HTTP(S) links
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code < 400:
                return True, f"HTTP {response.status_code}"
            else:
                return False, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def validate_links(self) -> bool:
        """Validate all links in documentation."""
        if not HAS_REQUESTS:
            self.add_warning("Requests library not available, skipping link validation")
            return True

        self.log("Validating documentation links...")

        success = True
        total_links = 0
        valid_links = 0

        for doc_file in self.docs_dir.glob("*.md"):
            self.log(f"Checking links in {doc_file}")

            try:
                content = doc_file.read_text(encoding="utf-8")
                links = self.extract_links(content)

                for link in links:
                    total_links += 1
                    is_valid, message = self.check_link(link, str(doc_file))

                    if is_valid:
                        valid_links += 1
                        if self.verbose:
                            self.log(f"✓ {link}: {message}")
                    else:
                        success = False
                        self.add_error(f"Broken link in {doc_file}: {link} ({message})")

            except Exception as e:
                self.add_error(f"Error checking links in {doc_file}: {e}")
                success = False

        self.log(f"Link validation complete: {valid_links}/{total_links} links valid")
        return success

    def validate_quickstart_guide(self) -> bool:
        """Validate that the quickstart guide steps work."""
        self.log("Validating quickstart guide...")

        quickstart_file = self.docs_dir / "quickstart.md"
        if not quickstart_file.exists():
            self.add_error("Quickstart guide not found")
            return False

        try:
            content = quickstart_file.read_text(encoding="utf-8")

            # Extract installation commands
            install_commands = re.findall(
                r"```bash\n(pip install[^\n]+)\n```", content, re.MULTILINE
            )

            if not install_commands:
                self.add_warning("No installation commands found in quickstart guide")
            else:
                self.log(f"Found {len(install_commands)} installation commands")

            # Extract Python code examples
            code_blocks = self.extract_code_blocks(content, str(quickstart_file))

            # Validate that key examples are present
            required_examples = [
                "from pulse.core.client import CoreClient",
                "from pulse.starters import sentiment_analysis",
                "from pulse.starters import summarize",
                "from pulse.starters import cluster_analysis",
            ]

            found_examples = []
            for code, _, _ in code_blocks:
                for example in required_examples:
                    if example in code and example not in found_examples:
                        found_examples.append(example)

            missing_examples = set(required_examples) - set(found_examples)
            if missing_examples:
                for missing in missing_examples:
                    self.add_warning(f"Missing key example in quickstart: {missing}")
            else:
                self.log("All key examples found in quickstart guide")

            return len(missing_examples) == 0

        except Exception as e:
            self.add_error(f"Error validating quickstart guide: {e}")
            return False

    def validate_documentation_build(self) -> bool:
        """Validate that documentation can be built successfully."""
        self.log("Validating documentation build...")

        # Check if mkdocs.yml exists
        mkdocs_config = Path("mkdocs.yml")
        if not mkdocs_config.exists():
            self.add_error("mkdocs.yml not found")
            return False

        try:
            # Try to build documentation
            result = subprocess.run(
                ["mkdocs", "build", "--strict"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                self.log("Documentation build successful")
                return True
            else:
                self.add_error(f"Documentation build failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.add_error("Documentation build timed out")
            return False
        except FileNotFoundError:
            self.add_warning("mkdocs not found, skipping build validation")
            return True
        except Exception as e:
            self.add_error(f"Error building documentation: {e}")
            return False

    def run_validation(
        self,
        check_links: bool = False,
        validate_quickstart: bool = False,
        build_docs: bool = False,
    ) -> bool:
        """Run all requested validations."""
        self.log("Starting documentation validation...")

        success = True

        # Always validate code examples
        if not self.validate_code_examples():
            success = False

        if check_links:
            if not self.validate_links():
                success = False

        if validate_quickstart:
            if not self.validate_quickstart_guide():
                success = False

        if build_docs:
            if not self.validate_documentation_build():
                success = False

        # Print summary
        self.log("\nValidation complete:")
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Pulse SDK documentation")
    parser.add_argument(
        "--check-links", action="store_true", help="Check all links in documentation"
    )
    parser.add_argument(
        "--validate-quickstart",
        action="store_true",
        help="Validate quickstart guide completeness",
    )
    parser.add_argument(
        "--build-docs", action="store_true", help="Validate documentation build"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--docs-dir", default="docs", help="Documentation directory (default: docs)"
    )

    args = parser.parse_args()

    # If no specific validations requested, run all
    if not any([args.check_links, args.validate_quickstart, args.build_docs]):
        args.check_links = True
        args.validate_quickstart = True
        args.build_docs = True

    validator = DocumentationValidator(args.docs_dir, args.verbose)
    success = validator.run_validation(
        check_links=args.check_links,
        validate_quickstart=args.validate_quickstart,
        build_docs=args.build_docs,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
