#!/usr/bin/env python3
"""
Cross-tool compatibility testing script for Pulse SDK.

This script tests installation and basic functionality across different
package managers (pip, pipenv, poetry, conda) to ensure compatibility.
"""

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any


class PackageManagerTester:
    """Test Pulse SDK installation across different package managers."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.test_dir = Path(tempfile.mkdtemp(prefix="pulse_sdk_test_"))
        print(f"Testing in: {self.test_dir}")

    def cleanup(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def run_command(self, cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
        """Run a command and return result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.test_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def test_basic_import(self, python_cmd: str = "python") -> bool:
        """Test basic Pulse SDK import."""
        test_script = """
import sys
try:
    import pulse
    from pulse.core.client import CoreClient
    from pulse.analysis.analyzer import Analyzer
    print(f"✅ Pulse SDK {pulse.__version__} imported successfully")
    print(f"Python: {sys.version}")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
"""

        script_path = self.test_dir / "test_import.py"
        script_path.write_text(test_script)

        result = self.run_command([python_cmd, str(script_path)])
        return result["success"]

    def test_pip(self) -> Dict[str, Any]:
        """Test pip installation."""
        print("\n🔧 Testing pip installation...")

        # Create virtual environment
        venv_dir = self.test_dir / "pip_venv"
        result = self.run_command([sys.executable, "-m", "venv", str(venv_dir)])
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to create venv",
                "details": result,
            }

        # Determine python and pip commands
        if os.name == "nt":  # Windows
            python_cmd = str(venv_dir / "Scripts" / "python.exe")
            pip_cmd = str(venv_dir / "Scripts" / "pip.exe")
        else:  # Unix-like
            python_cmd = str(venv_dir / "bin" / "python")
            pip_cmd = str(venv_dir / "bin" / "pip")

        # Upgrade pip
        result = self.run_command(
            [python_cmd, "-m", "pip", "install", "--upgrade", "pip"]
        )
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to upgrade pip",
                "details": result,
            }

        # Install pulse-sdk
        result = self.run_command([pip_cmd, "install", "pulse-sdk[minimal]"])
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to install pulse-sdk",
                "details": result,
            }

        # Test import
        import_success = self.test_basic_import(python_cmd)

        return {
            "success": import_success,
            "python_cmd": python_cmd,
            "pip_cmd": pip_cmd,
            "import_test": import_success,
        }

    def test_pipenv(self) -> Dict[str, Any]:
        """Test pipenv installation."""
        print("\n🔧 Testing pipenv installation...")

        # Check if pipenv is available
        result = self.run_command(["pipenv", "--version"])
        if not result["success"]:
            return {
                "success": False,
                "error": "pipenv not available",
                "details": result,
            }

        # Create pipenv project
        pipenv_dir = self.test_dir / "pipenv_test"
        pipenv_dir.mkdir()

        # Create Pipfile
        pipfile_content = """
[packages]
pulse-sdk = {extras = ["minimal"], version = "*"}

[dev-packages]

[requires]
python_version = "3.8"
"""
        (pipenv_dir / "Pipfile").write_text(pipfile_content)

        # Install dependencies
        result = self.run_command(["pipenv", "install"], cwd=pipenv_dir)
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to install with pipenv",
                "details": result,
            }

        # Test import
        result = self.run_command(
            [
                "pipenv",
                "run",
                "python",
                "-c",
                "import pulse; from pulse.core.client import CoreClient; print(f'✅ Pulse SDK {pulse.__version__} working')",
            ],
            cwd=pipenv_dir,
        )

        return {
            "success": result["success"],
            "import_test": result["success"],
            "details": result,
        }

    def test_poetry(self) -> Dict[str, Any]:
        """Test poetry installation."""
        print("\n🔧 Testing poetry installation...")

        # Check if poetry is available
        result = self.run_command(["poetry", "--version"])
        if not result["success"]:
            return {
                "success": False,
                "error": "poetry not available",
                "details": result,
            }

        # Create poetry project
        poetry_dir = self.test_dir / "poetry_test"
        poetry_dir.mkdir()

        # Initialize poetry project
        result = self.run_command(
            ["poetry", "init", "--no-interaction", "--name", "test-project"],
            cwd=poetry_dir,
        )
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to init poetry project",
                "details": result,
            }

        # Add pulse-sdk dependency
        result = self.run_command(
            ["poetry", "add", "pulse-sdk[minimal]"], cwd=poetry_dir
        )
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to add pulse-sdk with poetry",
                "details": result,
            }

        # Test import
        result = self.run_command(
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import pulse; from pulse.core.client import CoreClient; print(f'✅ Pulse SDK {pulse.__version__} working')",
            ],
            cwd=poetry_dir,
        )

        return {
            "success": result["success"],
            "import_test": result["success"],
            "details": result,
        }

    def test_conda(self) -> Dict[str, Any]:
        """Test conda installation."""
        print("\n🔧 Testing conda installation...")

        # Check if conda is available
        result = self.run_command(["conda", "--version"])
        if not result["success"]:
            return {"success": False, "error": "conda not available", "details": result}

        # Create conda environment
        env_name = "pulse_test_env"
        result = self.run_command(
            ["conda", "create", "-n", env_name, "python=3.9", "-y"]
        )
        if not result["success"]:
            return {
                "success": False,
                "error": "Failed to create conda env",
                "details": result,
            }

        # Install pulse-sdk via pip in conda env
        if os.name == "nt":  # Windows
            activate_cmd = ["conda", "run", "-n", env_name]
        else:  # Unix-like
            activate_cmd = ["conda", "run", "-n", env_name]

        result = self.run_command(
            activate_cmd + ["pip", "install", "pulse-sdk[minimal]"]
        )
        if not result["success"]:
            # Clean up environment
            self.run_command(["conda", "env", "remove", "-n", env_name, "-y"])
            return {
                "success": False,
                "error": "Failed to install pulse-sdk in conda",
                "details": result,
            }

        # Test import
        result = self.run_command(
            activate_cmd
            + [
                "python",
                "-c",
                "import pulse; from pulse.core.client import CoreClient; print(f'✅ Pulse SDK {pulse.__version__} working')",
            ]
        )

        # Clean up environment
        cleanup_result = self.run_command(
            ["conda", "env", "remove", "-n", env_name, "-y"]
        )

        return {
            "success": result["success"],
            "import_test": result["success"],
            "cleanup_success": cleanup_result["success"],
            "details": result,
        }

    def run_all_tests(self):
        """Run all package manager tests."""
        print("🚀 Starting Pulse SDK Package Manager Compatibility Tests")
        print("=" * 60)

        tests = [
            ("pip", self.test_pip),
            ("pipenv", self.test_pipenv),
            ("poetry", self.test_poetry),
            ("conda", self.test_conda),
        ]

        for name, test_func in tests:
            try:
                result = test_func()
                self.results[name] = result

                if result["success"]:
                    print(f"✅ {name}: PASSED")
                else:
                    print(f"❌ {name}: FAILED - {result.get('error', 'Unknown error')}")
                    if "details" in result and result["details"].get("stderr"):
                        print(
                            f"   Error details: {result['details']['stderr'][:200]}..."
                        )

            except Exception as e:
                print(f"❌ {name}: EXCEPTION - {str(e)}")
                self.results[name] = {"success": False, "error": str(e)}

        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        print("\nDetailed Results:")
        for name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {name:10} {status}")
            if not result["success"] and "error" in result:
                print(f"             Error: {result['error']}")

        if passed_tests == total_tests:
            print("\n🎉 All package managers are compatible with Pulse SDK!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} package manager(s) have issues.")
            print("   Check the error details above for troubleshooting.")


def main():
    """Main test function."""
    tester = PackageManagerTester()

    try:
        tester.run_all_tests()
    finally:
        tester.cleanup()

    # Exit with error code if any tests failed
    failed_tests = sum(1 for r in tester.results.values() if not r["success"])
    sys.exit(failed_tests)


if __name__ == "__main__":
    main()
