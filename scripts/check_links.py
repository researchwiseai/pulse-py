#!/usr/bin/env python3
"""
Link checker for Pulse SDK documentation.

This script checks all links in documentation files to ensure they are valid
and accessible.

Usage:
    python scripts/check_links.py [--file FILE] [--external] [--verbose]
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple
import threading

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LinkChecker:
    """Checks links in documentation files."""

    def __init__(
        self, verbose: bool = False, check_external: bool = True, timeout: int = 10
    ):
        self.verbose = verbose
        self.check_external = check_external
        self.timeout = timeout
        self.checked_urls = {}  # Cache for URL check results
        self.lock = threading.Lock()

        # Session for HTTP requests with retries
        if HAS_REQUESTS:
            self.session = requests.Session()
            # Add retry strategy
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

            # Set user agent
            self.session.headers.update(
                {"User-Agent": "Pulse-SDK-Link-Checker/1.0 (Documentation Validation)"}
            )

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")

    def extract_links(self, content: str, file_path: str) -> List[Tuple[str, int, str]]:
        """Extract all links from markdown content with line numbers."""
        links = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Markdown links: [text](url)
            markdown_links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", line)
            for text, url in markdown_links:
                links.append((url.strip(), line_num, f"markdown link: [{text}]({url})"))

            # Reference links: [text]: url
            ref_links = re.findall(r"^\[([^\]]+)\]:\s*(.+)$", line)
            for text, url in ref_links:
                links.append(
                    (url.strip(), line_num, f"reference link: [{text}]: {url}")
                )

            # HTML links: <a href="url">
            html_links = re.findall(
                r'<a[^>]+href=["\']([^"\']+)["\']', line, re.IGNORECASE
            )
            for url in html_links:
                links.append((url.strip(), line_num, f'HTML link: href="{url}"'))

            # Direct URLs: http(s)://...
            direct_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', line)
            for url in direct_urls:
                links.append((url.strip(), line_num, f"direct URL: {url}"))

        return links

    def is_external_url(self, url: str) -> bool:
        """Check if URL is external (HTTP/HTTPS)."""
        return url.startswith(("http://", "https://"))

    def is_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped."""
        skip_patterns = [
            r"example\.com",
            r"localhost",
            r"127\.0\.0\.1",
            r"0\.0\.0\.0",
            r"your_.*",
            r"<.*>",
            r"\$\{.*\}",
            r"mailto:.*@example\.com",
            r"mailto:dev@researchwiseai\.com",  # Skip our own email
            r"#.*",  # Skip anchor-only links
            r"javascript:",
            r"data:",
        ]

        for pattern in skip_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def check_local_file(self, url: str, base_path: Path) -> Tuple[bool, str]:
        """Check if a local file exists."""
        # Handle relative paths
        if url.startswith("/"):
            # Absolute path from project root
            file_path = Path(url[1:])  # Remove leading slash
        else:
            # Relative to current file
            file_path = base_path.parent / url

        # Handle common documentation file extensions
        if not file_path.suffix:
            # Try adding .md extension
            md_path = file_path.with_suffix(".md")
            if md_path.exists():
                return True, f"Found as {md_path}"
            # Try as directory with index
            index_path = file_path / "index.md"
            if index_path.exists():
                return True, f"Found as {index_path}"

        if file_path.exists():
            return True, f"File exists: {file_path}"
        else:
            return False, f"File not found: {file_path}"

    def check_http_url(self, url: str) -> Tuple[bool, str]:
        """Check if HTTP URL is accessible."""
        if not HAS_REQUESTS:
            return True, "Requests not available, skipping HTTP check"

        # Check cache first
        with self.lock:
            if url in self.checked_urls:
                return self.checked_urls[url]

        try:
            # Use HEAD request first (faster)
            response = self.session.head(
                url, timeout=self.timeout, allow_redirects=True
            )

            # Some servers don't support HEAD, try GET if HEAD fails
            if response.status_code == 405:  # Method Not Allowed
                response = self.session.get(
                    url, timeout=self.timeout, allow_redirects=True
                )

            success = response.status_code < 400
            message = f"HTTP {response.status_code}"

            # Cache result
            with self.lock:
                self.checked_urls[url] = (success, message)

            return success, message

        except requests.exceptions.Timeout:
            result = (False, "Timeout")
        except requests.exceptions.ConnectionError:
            result = (False, "Connection error")
        except requests.exceptions.TooManyRedirects:
            result = (False, "Too many redirects")
        except requests.exceptions.RequestException as e:
            result = (False, f"Request error: {str(e)}")
        except Exception as e:
            result = (False, f"Unexpected error: {str(e)}")

        # Cache result
        with self.lock:
            self.checked_urls[url] = result

        return result

    def check_link(self, url: str, file_path: Path) -> Tuple[bool, str]:
        """Check a single link."""
        # Skip certain URLs
        if self.is_skip_url(url):
            return True, "Skipped (example/placeholder URL)"

        # Handle anchor links
        if url.startswith("#"):
            return True, "Anchor link (not validated)"

        # Handle external URLs
        if self.is_external_url(url):
            if not self.check_external:
                return True, "External link (not checked)"
            return self.check_http_url(url)

        # Handle local files
        return self.check_local_file(url, file_path)

    def check_file_links(
        self, file_path: Path
    ) -> Tuple[int, int, List[Tuple[str, int, str, str]]]:
        """Check all links in a single file."""
        self.log(f"Checking links in {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
            links = self.extract_links(content, str(file_path))

            if not links:
                self.log(f"No links found in {file_path}")
                return 0, 0, []

            valid_count = 0
            total_count = len(links)
            failures = []

            for url, line_num, context in links:
                is_valid, message = self.check_link(url, file_path)

                if is_valid:
                    valid_count += 1
                    if self.verbose:
                        self.log(f"✓ {file_path}:{line_num} {url} - {message}")
                else:
                    failure = (url, line_num, context, message)
                    failures.append(failure)
                    self.log(f"✗ {file_path}:{line_num} {url} - {message}", "ERROR")

                # Small delay to be respectful to servers
                if self.is_external_url(url) and self.check_external:
                    time.sleep(0.1)

            self.log(f"Completed {file_path}: {valid_count}/{total_count} links valid")
            return valid_count, total_count, failures

        except Exception as e:
            self.log(f"Error processing {file_path}: {e}", "ERROR")
            return 0, 0, [(str(file_path), 0, "file processing", str(e))]

    def check_all_links(self, docs_dir: Path) -> bool:
        """Check links in all documentation files."""
        self.log("Starting link validation...")

        total_valid = 0
        total_links = 0
        all_failures = []

        # Get all markdown files
        md_files = list(docs_dir.glob("*.md"))

        if not md_files:
            self.log(f"No markdown files found in {docs_dir}")
            return True

        # Check files sequentially to avoid overwhelming servers
        for file_path in md_files:
            valid, total, failures = self.check_file_links(file_path)
            total_valid += valid
            total_links += total
            all_failures.extend(
                [
                    (str(file_path), line, context, msg)
                    for url, line, context, msg in failures
                ]
            )

        # Print summary
        self.log("\nLink Check Summary:")
        self.log(f"  Total links: {total_links}")
        self.log(f"  Valid links: {total_valid}")
        self.log(f"  Failed links: {len(all_failures)}")

        if all_failures:
            self.log("\nFailures:")
            for file_path, line_num, context, message in all_failures:
                self.log(f"  - {file_path}:{line_num} {context} - {message}")

        return len(all_failures) == 0

    def check_single_file(self, file_path: Path) -> bool:
        """Check links in a single file."""
        valid, total, failures = self.check_file_links(file_path)
        return len(failures) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check links in Pulse SDK documentation"
    )
    parser.add_argument("--file", type=Path, help="Check links in a specific file")
    parser.add_argument(
        "--external",
        action="store_true",
        default=True,
        help="Check external HTTP(S) links (default: True)",
    )
    parser.add_argument(
        "--no-external", action="store_true", help="Skip external HTTP(S) links"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation directory (default: docs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP request timeout in seconds (default: 10)",
    )

    args = parser.parse_args()

    if not HAS_REQUESTS:
        print(
            "Warning: requests library not available, external link checking disabled"
        )

    check_external = args.external and not args.no_external and HAS_REQUESTS

    link_checker = LinkChecker(
        verbose=args.verbose, check_external=check_external, timeout=args.timeout
    )

    if args.file:
        if not args.file.exists():
            print(f"Error: File {args.file} does not exist")
            sys.exit(1)
        success = link_checker.check_single_file(args.file)
    else:
        if not args.docs_dir.exists():
            print(f"Error: Documentation directory {args.docs_dir} does not exist")
            sys.exit(1)
        success = link_checker.check_all_links(args.docs_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
