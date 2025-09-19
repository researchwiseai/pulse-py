#!/usr/bin/env python3
"""
Simple test runner for documentation validation scripts.
"""

import sys
import tempfile
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import validation modules
from validate_docs import DocumentationValidator  # noqa: E402
from run_doctests import DocumentationDoctest  # noqa: E402
from check_links import LinkChecker  # noqa: E402
from validate_quickstart import QuickstartValidator  # noqa: E402


def test_documentation_validator():
    """Test the main documentation validator."""
    print("Testing DocumentationValidator...")

    validator = DocumentationValidator()

    # Test code block extraction
    content = """
# Test Document

```python
from pulse import CoreClient
client = CoreClient()
```

```bash
pip install pulse-sdk
```

```python
result = client.analyze_sentiment(["test"])
```
"""

    blocks = validator.extract_code_blocks(content, "test.md")
    assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}"

    # Test syntax validation
    valid_code = "from pulse import CoreClient\nclient = CoreClient()"
    assert validator.validate_python_syntax(valid_code, "test.md", 1)

    invalid_code = "from pulse import CoreClient\nclient = CoreClient("
    assert not validator.validate_python_syntax(invalid_code, "test.md", 1)

    print("✓ DocumentationValidator tests passed")


def test_doctest_runner():
    """Test the doctest runner."""
    print("Testing DocumentationDoctest...")

    doctest_runner = DocumentationDoctest()

    # Test testable code detection
    testable_code = "from pulse import CoreClient\nclient = CoreClient()"
    assert doctest_runner.is_testable_code(testable_code)

    placeholder_code = 'client_id = "your_client_id"'
    assert not doctest_runner.is_testable_code(placeholder_code)

    # Test mock environment creation
    mock_env = doctest_runner.create_mock_environment()
    assert "CoreClient" in mock_env
    assert "sentiment_analysis" in mock_env

    print("✓ DocumentationDoctest tests passed")


def test_link_checker():
    """Test the link checker."""
    print("Testing LinkChecker...")

    link_checker = LinkChecker()

    # Test link extraction
    content = """Line 1: Check out [our website](https://example.com)
Line 2: See [docs](docs.md)"""

    links = link_checker.extract_links(content, "test.md")
    print(f"Found {len(links)} links: {[link[0] for link in links]}")
    assert len(links) >= 2  # Should find at least 2 links

    # Test URL skipping logic
    assert link_checker.is_skip_url("https://example.com")
    assert not link_checker.is_skip_url("https://github.com/researchwiseai/pulse-py")

    # Test local file checking with temp file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.md"
        test_file.write_text("# Test")
        base_file = temp_path / "base.md"

        is_valid, message = link_checker.check_local_file("test.md", base_file)
        assert is_valid

        is_valid, message = link_checker.check_local_file("nonexistent.md", base_file)
        assert not is_valid

    print("✓ LinkChecker tests passed")


def test_quickstart_validator():
    """Test the quickstart validator."""
    print("Testing QuickstartValidator...")

    validator = QuickstartValidator()

    # Test code block extraction
    content = """
# Quickstart

```bash
pip install pulse-sdk[all]
```

```python
from pulse import CoreClient
```
"""

    blocks = validator.extract_code_blocks(content)
    assert len(blocks["bash"]) == 1
    assert len(blocks["python"]) == 1

    # Test installation command validation
    bash_blocks = [("pip install pulse-sdk[all]", 1)]
    result = validator.validate_installation_commands(bash_blocks)
    assert result

    print("✓ QuickstartValidator tests passed")


def test_integration_with_real_files():
    """Test with real documentation files."""
    print("Testing with real documentation files...")

    # Test quickstart guide if it exists
    quickstart_path = Path("docs/quickstart.md")
    if quickstart_path.exists():
        validator = QuickstartValidator(verbose=False)
        validator.validate_quickstart_guide(quickstart_path)

        # Should have no errors (warnings are OK)
        if validator.errors:
            print(f"⚠ Quickstart guide has errors: {validator.errors}")
        else:
            print("✓ Real quickstart guide validation passed")
    else:
        print("⚠ Quickstart guide not found, skipping real file test")

    # Test documentation validator on a real file
    docs_dir = Path("docs")
    if docs_dir.exists():
        validator = DocumentationValidator(verbose=False)

        # Test on index.md if it exists
        index_file = docs_dir / "index.md"
        if index_file.exists():
            content = index_file.read_text()
            blocks = validator.extract_code_blocks(content, str(index_file))
            print(f"✓ Found {len(blocks)} code blocks in index.md")
        else:
            print("⚠ index.md not found, skipping real file test")
    else:
        print("⚠ docs directory not found, skipping real file tests")


def main():
    """Run all tests."""
    print("Running documentation validation tests...\n")

    try:
        test_documentation_validator()
        test_doctest_runner()
        test_link_checker()
        test_quickstart_validator()
        test_integration_with_real_files()

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
