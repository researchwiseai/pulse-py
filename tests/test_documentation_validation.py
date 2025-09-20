"""
Tests for documentation validation scripts.

This module tests the documentation validation tools to ensure they work correctly
and catch common documentation issues.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from validate_docs import DocumentationValidator
    from run_doctests import DocumentationDoctest
    from check_links import LinkChecker
    from validate_quickstart import QuickstartValidator

    HAS_VALIDATION_SCRIPTS = True
except ImportError:
    HAS_VALIDATION_SCRIPTS = False


@pytest.mark.skipif(
    not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
)
class TestDocumentationValidator:
    """Test the main documentation validator."""

    def test_extract_code_blocks(self):
        """Test extraction of Python code blocks from markdown."""
        validator = DocumentationValidator()

        content = """
# Test Document

Some text here.

```python
from pulse import CoreClient
client = CoreClient()
```

```bash
pip install pulse-sdk
```

```python
# Another Python block
result = client.analyze_sentiment(["test"])
```
"""

        blocks = validator.extract_code_blocks(content, "test.md")
        assert len(blocks) == 2

        # Check first block
        code1, file_path1, line1 = blocks[0]
        assert "from pulse import CoreClient" in code1
        assert "client = CoreClient()" in code1
        assert file_path1 == "test.md"
        assert line1 == 6  # Line where code block starts

        # Check second block
        code2, file_path2, line2 = blocks[1]
        assert "result = client.analyze_sentiment" in code2
        assert file_path2 == "test.md"

    def test_validate_python_syntax_valid(self):
        """Test validation of valid Python syntax."""
        validator = DocumentationValidator()

        valid_code = """
from pulse import CoreClient
client = CoreClient()
result = client.analyze_sentiment(["test"])
"""

        assert validator.validate_python_syntax(valid_code, "test.md", 1)
        assert len(validator.errors) == 0

    def test_validate_python_syntax_invalid(self):
        """Test validation of invalid Python syntax."""
        validator = DocumentationValidator()

        invalid_code = """
from pulse import CoreClient
client = CoreClient(
# Missing closing parenthesis
"""

        assert not validator.validate_python_syntax(invalid_code, "test.md", 1)
        assert len(validator.errors) == 1
        assert "Syntax error" in validator.errors[0]

    def test_extract_links(self):
        """Test extraction of links from markdown content."""
        validator = DocumentationValidator()

        content = """
# Test Document

Check out [our website](https://example.com) for more info.

Also see the [installation guide](installation.md).

Reference link: [GitHub]: https://github.com/example/repo

Direct URL: https://docs.example.com

HTML link: <a href="https://api.example.com">API Docs</a>
"""

        links = validator.extract_links(content)
        expected_links = [
            "https://example.com",
            "installation.md",
            "https://github.com/example/repo",
            "https://docs.example.com",
            "https://api.example.com",
        ]

        assert len(links) == len(expected_links)
        for expected in expected_links:
            assert expected in links


@pytest.mark.skipif(
    not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
)
class TestDocumentationDoctest:
    """Test the doctest runner for documentation."""

    def test_extract_testable_code_blocks(self):
        """Test extraction of testable code blocks."""
        doctest_runner = DocumentationDoctest()

        content = """
# Test Document

```python
from pulse import CoreClient
client = CoreClient()
```

```python
# This should be skipped - has placeholder
client_id = "your_client_id"
```

```bash
# This should be skipped - not Python
pip install pulse-sdk
```

```python
# This should be included
result = client.analyze_sentiment(["test"])
print(result)
```
"""

        blocks = doctest_runner.extract_testable_code_blocks(content, "test.md")

        # Should have 2 testable blocks (skipping placeholder and bash)
        assert len(blocks) == 2

        # Check that placeholder code is filtered out
        for code, line, file_path in blocks:
            assert "your_client_id" not in code
            assert "pip install" not in code

    def test_is_testable_code(self):
        """Test the testable code detection logic."""
        doctest_runner = DocumentationDoctest()

        # Should be testable
        testable_code = """
from pulse import CoreClient
client = CoreClient()
result = client.analyze_sentiment(["test"])
"""
        assert doctest_runner.is_testable_code(testable_code)

        # Should not be testable - has placeholder
        placeholder_code = """
client_id = "your_client_id"
client_secret = "your_client_secret"
"""
        assert not doctest_runner.is_testable_code(placeholder_code)

        # Should not be testable - shell command
        shell_code = "pip install pulse-sdk"
        assert not doctest_runner.is_testable_code(shell_code)

    def test_create_mock_environment(self):
        """Test creation of mock environment for testing."""
        doctest_runner = DocumentationDoctest()

        mock_env = doctest_runner.create_mock_environment()

        # Check that required mocks are present
        assert "CoreClient" in mock_env
        assert "sentiment_analysis" in mock_env
        assert "summarize" in mock_env
        assert "cluster_analysis" in mock_env

        # Test that mocks work
        client = mock_env["CoreClient"]()
        result = client.analyze_sentiment(["test"])
        assert hasattr(result, "results")

    @patch("builtins.exec")
    def test_run_code_block_success(self, mock_exec):
        """Test successful execution of a code block."""
        doctest_runner = DocumentationDoctest()

        # Mock successful execution
        mock_exec.return_value = None

        code = """
from pulse import CoreClient
client = CoreClient()
"""

        success = doctest_runner.run_code_block(code, "test.md", 1)
        assert success
        assert len(doctest_runner.failures) == 0
        assert len(doctest_runner.successes) == 1

    @patch("builtins.exec")
    def test_run_code_block_failure(self, mock_exec):
        """Test failed execution of a code block."""
        doctest_runner = DocumentationDoctest()

        # Mock execution failure with syntax error (should fail)
        mock_exec.side_effect = SyntaxError("invalid syntax")

        code = """
print(undefined_var
"""

        success = doctest_runner.run_code_block(code, "test.md", 1)
        assert not success
        assert len(doctest_runner.failures) == 1
        assert len(doctest_runner.successes) == 0


@pytest.mark.skipif(
    not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
)
class TestLinkChecker:
    """Test the link checker."""

    def test_extract_links(self):
        """Test extraction of links with line numbers."""
        link_checker = LinkChecker()

        content = """Line 1: Check out [our website](https://example.com)
Line 2: See [docs](docs.md)
Line 3: Reference [GitHub]: https://github.com/example/repo
Line 4: Direct URL https://api.example.com
Line 5: HTML <a href="https://test.com">link</a>"""

        links = link_checker.extract_links(content, "test.md")

        # Should find 5 links
        assert len(links) == 5

        # Check that line numbers are correct
        urls_and_lines = [(url, line) for url, line, context in links]

        assert ("https://example.com", 1) in urls_and_lines
        assert ("docs.md", 2) in urls_and_lines
        assert ("https://github.com/example/repo", 3) in urls_and_lines
        assert ("https://api.example.com", 4) in urls_and_lines
        assert ("https://test.com", 5) in urls_and_lines

    def test_is_skip_url(self):
        """Test URL skipping logic."""
        link_checker = LinkChecker()

        # Should skip these URLs
        skip_urls = [
            "https://example.com",
            "http://localhost:3000",
            "mailto:test@example.com",
            "#anchor-link",
            "javascript:void(0)",
            "your_api_key",
        ]

        for url in skip_urls:
            assert link_checker.is_skip_url(url), f"Should skip {url}"

        # Should not skip these URLs
        valid_urls = [
            "https://github.com/researchwiseai/pulse-py",
            "https://docs.python.org",
            "installation.md",
            "/docs/quickstart.md",
        ]

        for url in valid_urls:
            assert not link_checker.is_skip_url(url), f"Should not skip {url}"

    def test_check_local_file_exists(self):
        """Test checking local file existence."""
        link_checker = LinkChecker()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a test file
            test_file = temp_path / "test.md"
            test_file.write_text("# Test")

            # Create base file for relative path resolution
            base_file = temp_path / "base.md"
            base_file.write_text("# Base")

            # Test existing file
            is_valid, message = link_checker.check_local_file("test.md", base_file)
            assert is_valid
            assert "exists" in message.lower()

            # Test non-existing file
            is_valid, message = link_checker.check_local_file(
                "nonexistent.md", base_file
            )
            assert not is_valid
            assert "not found" in message.lower()

    @pytest.mark.skipif(
        not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
    )
    def test_check_http_url_success(self):
        """Test successful HTTP URL checking."""
        from scripts.check_links import HAS_REQUESTS

        if not HAS_REQUESTS:
            pytest.skip("requests not available")

        with patch("requests.Session.head") as mock_head:
            link_checker = LinkChecker()

            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_head.return_value = mock_response

            is_valid, message = link_checker.check_http_url("https://example.com")
            assert is_valid
            assert "200" in message

    @pytest.mark.skipif(
        not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
    )
    def test_check_http_url_failure(self):
        """Test failed HTTP URL checking."""
        from scripts.check_links import HAS_REQUESTS

        if not HAS_REQUESTS:
            pytest.skip("requests not available")

        with patch("requests.Session.head") as mock_head:
            link_checker = LinkChecker()

            # Mock failed response
            mock_response = Mock()
            mock_response.status_code = 404
            mock_head.return_value = mock_response

            is_valid, message = link_checker.check_http_url(
                "https://example.com/notfound"
            )
            assert not is_valid
            assert "404" in message


@pytest.mark.skipif(
    not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
)
class TestQuickstartValidator:
    """Test the quickstart guide validator."""

    def test_extract_code_blocks(self):
        """Test extraction of code blocks by language."""
        validator = QuickstartValidator()

        content = """
# Quickstart

Install the SDK:

```bash
pip install pulse-sdk[all]
```

Then use it:

```python
from pulse import CoreClient
client = CoreClient()
```

Another shell command:

```shell
export PULSE_CLIENT_ID="test"
```
"""

        blocks = validator.extract_code_blocks(content)

        assert len(blocks["bash"]) == 1
        assert len(blocks["python"]) == 1
        assert len(blocks["shell"]) == 1

        # Check content
        assert "pip install pulse-sdk[all]" in blocks["bash"][0][0]
        assert "from pulse import CoreClient" in blocks["python"][0][0]
        assert "export PULSE_CLIENT_ID" in blocks["shell"][0][0]

    def test_validate_installation_commands(self):
        """Test validation of installation commands."""
        validator = QuickstartValidator()

        bash_blocks = [
            ("pip install pulse-sdk[all]", 1),
            ("pip install pulse-sdk[minimal]", 5),
            ("pip install pulse-sdk[dev]", 10),
        ]

        result = validator.validate_installation_commands(bash_blocks)
        assert result
        assert len(validator.errors) == 0

    def test_validate_python_examples(self):
        """Test validation of Python examples."""
        validator = QuickstartValidator()

        python_blocks = [
            ("from pulse.core.client import CoreClient\nclient = CoreClient()", 1),
            (
                "from pulse.starters import sentiment_analysis\n"
                "result = sentiment_analysis(['test'])",
                5,
            ),
            ("from pulse.starters import summarize\nsummary = summarize(['text'])", 10),
        ]

        result = validator.validate_python_examples(python_blocks)
        assert result
        # Should have some warnings for missing examples, but not errors

    def test_validate_structure_and_content(self):
        """Test validation of quickstart structure."""
        validator = QuickstartValidator()

        content = """
# Quick Start Guide

## Installation

Install the SDK in under 5 minutes.

## Authentication Setup

Configure your credentials.

## Troubleshooting

Common issues and solutions.

### Customer Feedback Analysis
### Sentiment Analysis
### Theme Analysis
"""

        result = validator.validate_structure_and_content(content)
        assert result

    def test_validate_error_handling_guidance(self):
        """Test validation of error handling guidance."""
        validator = QuickstartValidator()

        content = """
# Troubleshooting

## ImportError
If you get an ImportError, try reinstalling.

## PulseAPIError: 401 Unauthorized
Check your credentials.

## Connection timeout
Check your network connection.

Problem: Installation fails
Solution: Use a virtual environment.
"""

        result = validator.validate_error_handling_guidance(content)
        assert result


@pytest.mark.skipif(
    not HAS_VALIDATION_SCRIPTS, reason="Validation scripts not available"
)
def test_integration_with_real_quickstart():
    """Integration test with the actual quickstart guide."""
    quickstart_path = Path("docs/quickstart.md")

    if not quickstart_path.exists():
        pytest.skip("Quickstart guide not found")

    validator = QuickstartValidator(verbose=False)

    # This should pass with the real quickstart guide
    # If it fails, it indicates issues with the actual documentation
    validator.validate_quickstart_guide(quickstart_path)

    # Print any issues for debugging
    if validator.errors:
        print("Errors found in quickstart guide:")
        for error in validator.errors:
            print(f"  - {error}")

    if validator.warnings:
        print("Warnings found in quickstart guide:")
        for warning in validator.warnings:
            print(f"  - {warning}")

    # Allow some warnings, but no errors
    assert (
        len(validator.errors) == 0
    ), f"Quickstart guide has errors: {validator.errors}"


if __name__ == "__main__":
    pytest.main([__file__])
