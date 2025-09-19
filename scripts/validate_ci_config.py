#!/usr/bin/env python3
"""
Validate CI/CD configuration for security and quality checks.

This script validates that all required tools and configurations are properly set up
for the CI/CD pipeline security and quality assurance.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def check_tool_availability() -> Dict[str, bool]:
    """Check if required tools are available."""
    tools = {
        "python": ["python", "--version"],
        "pip": ["pip", "--version"],
        "pytest": ["pytest", "--version"],
        "black": ["black", "--version"],
        "ruff": ["ruff", "--version"],
        "bandit": ["bandit", "--version"],
        "pip-audit": ["pip-audit", "--version"],
        "pre-commit": ["pre-commit", "--version"],
    }

    results = {}
    for tool, cmd in tools.items():
        exit_code, stdout, stderr = run_command(cmd)
        results[tool] = exit_code == 0
        if exit_code == 0:
            print(f"✅ {tool}: Available")
        else:
            print(f"❌ {tool}: Not available - {stderr}")

    return results


def check_configuration_files() -> Dict[str, bool]:
    """Check if required configuration files exist and are valid."""
    files = {
        "pyproject.toml": Path("pyproject.toml"),
        ".pre-commit-config.yaml": Path(".pre-commit-config.yaml"),
        "ci.yml": Path(".github/workflows/ci.yml"),
        "security.yml": Path(".github/workflows/security.yml"),
    }

    results = {}
    for name, path in files.items():
        exists = path.exists()
        results[name] = exists
        if exists:
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Missing")

    return results


def check_bandit_config() -> bool:
    """Check Bandit configuration in pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("❌ Cannot check pyproject.toml - tomllib/tomli not available")
            return False

    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        bandit_config = config.get("tool", {}).get("bandit", {})
        if bandit_config:
            print("✅ Bandit configuration found in pyproject.toml")
            print(f"   - Excluded dirs: {bandit_config.get('exclude_dirs', [])}")
            print(f"   - Skipped tests: {bandit_config.get('skips', [])}")
            return True
        else:
            print("❌ Bandit configuration missing from pyproject.toml")
            return False
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return False


def check_coverage_config() -> bool:
    """Check coverage configuration in pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return False

    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        coverage_config = config.get("tool", {}).get("coverage", {})
        if coverage_config:
            report_config = coverage_config.get("report", {})
            fail_under = report_config.get("fail_under", 0)
            print(f"✅ Coverage configuration found - threshold: {fail_under}%")
            return fail_under >= 90
        else:
            print("❌ Coverage configuration missing from pyproject.toml")
            return False
    except Exception as e:
        print(f"❌ Error reading coverage config: {e}")
        return False


def test_security_tools() -> Dict[str, bool]:
    """Test security tools with sample code."""
    results = {}

    # Test Bandit
    print("\n🔍 Testing Bandit...")
    exit_code, stdout, stderr = run_command(
        ["bandit", "-r", "pulse", "--format", "json", "--quiet"]
    )
    results["bandit"] = exit_code in [0, 1]  # 0 = no issues, 1 = issues found
    if results["bandit"]:
        print("✅ Bandit scan completed")
    else:
        print(f"❌ Bandit scan failed: {stderr}")

    # Test pip-audit
    print("\n🔍 Testing pip-audit...")
    exit_code, stdout, stderr = run_command(["pip-audit", "--format", "json"])
    results["pip-audit"] = exit_code in [0, 1]  # 0 = no vulns, 1 = vulns found
    if results["pip-audit"]:
        print("✅ pip-audit scan completed")
    else:
        print(f"❌ pip-audit scan failed: {stderr}")

    return results


def test_code_quality_tools() -> Dict[str, bool]:
    """Test code quality tools."""
    results = {}

    # Test Black
    print("\n🎨 Testing Black...")
    exit_code, stdout, stderr = run_command(["black", "--check", "--diff", "pulse"])
    results["black"] = exit_code == 0
    if results["black"]:
        print("✅ Code formatting is correct")
    else:
        print("⚠️  Code formatting issues found (run 'black .' to fix)")

    # Test Ruff
    print("\n🔍 Testing Ruff...")
    exit_code, stdout, stderr = run_command(["ruff", "check", "pulse", "tests"])
    results["ruff"] = exit_code == 0
    if results["ruff"]:
        print("✅ No linting issues found")
    else:
        print("⚠️  Linting issues found (run 'ruff check --fix' to fix)")

    return results


def main():
    """Main validation function."""
    print("🔍 Validating CI/CD Security and Quality Configuration\n")

    all_passed = True

    # Check tool availability
    print("📋 Checking tool availability...")
    tool_results = check_tool_availability()
    if not all(tool_results.values()):
        print("❌ Some required tools are missing")
        all_passed = False

    print("\n📁 Checking configuration files...")
    config_results = check_configuration_files()
    if not all(config_results.values()):
        print("❌ Some required configuration files are missing")
        all_passed = False

    print("\n⚙️  Checking Bandit configuration...")
    bandit_ok = check_bandit_config()
    if not bandit_ok:
        all_passed = False

    print("\n📊 Checking coverage configuration...")
    coverage_ok = check_coverage_config()
    if not coverage_ok:
        all_passed = False

    # Test tools if available
    if tool_results.get("bandit") and tool_results.get("pip-audit"):
        print("\n🛡️  Testing security tools...")
        security_results = test_security_tools()
        if not all(security_results.values()):
            print("⚠️  Some security tools had issues")

    if tool_results.get("black") and tool_results.get("ruff"):
        print("\n🎯 Testing code quality tools...")
        test_code_quality_tools()
        # Don't fail on quality issues, just warn

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All CI/CD security and quality checks are properly configured!")
        print("\nNext steps:")
        print("1. Run 'pre-commit install' to set up git hooks")
        print("2. Run 'pre-commit run --all-files' to test hooks")
        print("3. Commit changes and verify CI pipeline runs successfully")
        return 0
    else:
        print("❌ Some configuration issues found. Please fix them before proceeding.")
        print("\nRefer to docs/ci-cd-security.md for detailed setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
