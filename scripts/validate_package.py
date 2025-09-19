#!/usr/bin/env python3
"""
Package validation script for Pulse SDK.

This script validates the package build, metadata, and distribution
to ensure it follows Python packaging best practices.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("ERROR: tomllib/tomli not available. Install with: pip install tomli")
        sys.exit(1)


class PackageValidator:
    """Validate Python package build and metadata."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dist_dir = self.project_root / "dist"
        self.results: Dict[str, Any] = {}

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run a command and return result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=120,
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

    def validate_pyproject_toml(self) -> Dict[str, Any]:
        """Validate pyproject.toml metadata."""
        print("🔍 Validating pyproject.toml metadata...")

        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            return {"success": False, "error": "pyproject.toml not found"}

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse pyproject.toml: {e}"}

        issues = []
        project = data.get("project", {})

        # Check required fields
        required_fields = ["name", "version", "description", "authors"]
        for field in required_fields:
            if not project.get(field):
                issues.append(f"Missing required field: {field}")

        # Check version format (basic check)
        version = project.get("version", "")
        if (
            version
            and not version.replace(".", "")
            .replace("-", "")
            .replace("+", "")
            .replace("a", "")
            .replace("b", "")
            .replace("rc", "")
            .isalnum()
        ):
            issues.append(f"Invalid version format: {version}")

        # Check license (modern SPDX format or legacy format)
        license_info = project.get("license")
        if not license_info:
            issues.append("Missing license information")

        # Check URLs
        urls = project.get("urls", {})
        required_urls = ["Homepage", "Repository", "Documentation"]
        for url_name in required_urls:
            if url_name not in urls:
                issues.append(f"Missing URL: {url_name}")

        # Check classifiers
        classifiers = project.get("classifiers", [])
        if not classifiers:
            issues.append("No classifiers specified")
        else:
            # Check for important classifier categories (License is optional with SPDX)
            categories = {
                "Development Status": False,
                "Programming Language": False,
                "Operating System": False,
            }

            for classifier in classifiers:
                for category in categories:
                    if classifier.startswith(category):
                        categories[category] = True

            for category, found in categories.items():
                if not found:
                    issues.append(f"Missing classifier category: {category}")

        # Check dependencies
        dependencies = project.get("dependencies", [])
        if not dependencies:
            issues.append("No dependencies specified")

        return {"success": len(issues) == 0, "issues": issues, "metadata": project}

    def build_package(self) -> Dict[str, Any]:
        """Build the package using build tool."""
        print("🔨 Building package...")

        # Clean previous builds
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)

        # Build package
        result = self.run_command([sys.executable, "-m", "build"])

        if not result["success"]:
            return {"success": False, "error": "Build failed", "details": result}

        # Check build artifacts
        if not self.dist_dir.exists():
            return {"success": False, "error": "dist directory not created"}

        artifacts = list(self.dist_dir.glob("*"))
        sdist_files = [f for f in artifacts if f.suffix == ".tar.gz"]
        wheel_files = [f for f in artifacts if f.suffix == ".whl"]

        return {
            "success": True,
            "artifacts": [str(f.name) for f in artifacts],
            "sdist_files": [str(f.name) for f in sdist_files],
            "wheel_files": [str(f.name) for f in wheel_files],
            "total_artifacts": len(artifacts),
        }

    def validate_wheel(self) -> Dict[str, Any]:
        """Validate wheel file structure and metadata."""
        print("🎯 Validating wheel file...")

        wheel_files = list(self.dist_dir.glob("*.whl"))
        if not wheel_files:
            return {"success": False, "error": "No wheel file found"}

        wheel_path = wheel_files[0]
        issues = []

        try:
            with zipfile.ZipFile(wheel_path, "r") as wheel:
                files = wheel.namelist()

                # Check for required files
                has_metadata = any(f.endswith(".dist-info/METADATA") for f in files)
                has_wheel_info = any(f.endswith(".dist-info/WHEEL") for f in files)
                has_py_typed = any(f.endswith("py.typed") for f in files)

                if not has_metadata:
                    issues.append("Missing METADATA file in wheel")

                if not has_wheel_info:
                    issues.append("Missing WHEEL file in wheel")

                if not has_py_typed:
                    issues.append("Missing py.typed file (type information)")

                # Check package structure
                package_files = [
                    f for f in files if f.startswith("pulse/") and f.endswith(".py")
                ]
                if not package_files:
                    issues.append("No Python package files found in wheel")

                # Read and validate METADATA
                if has_metadata:
                    metadata_file = next(
                        f for f in files if f.endswith(".dist-info/METADATA")
                    )
                    metadata_content = wheel.read(metadata_file).decode("utf-8")

                    # Check for required metadata fields
                    # (Author and License are optional in modern format)
                    required_metadata = [
                        "Metadata-Version:",
                        "Name:",
                        "Version:",
                        "Summary:",
                    ]

                    for field in required_metadata:
                        if field not in metadata_content:
                            issues.append(f"Missing metadata field: {field}")

        except Exception as e:
            return {"success": False, "error": f"Failed to validate wheel: {e}"}

        return {
            "success": len(issues) == 0,
            "issues": issues,
            "wheel_file": str(wheel_path.name),
        }

    def validate_sdist(self) -> Dict[str, Any]:
        """Validate source distribution."""
        print("📦 Validating source distribution...")

        sdist_files = list(self.dist_dir.glob("*.tar.gz"))
        if not sdist_files:
            return {"success": False, "error": "No source distribution found"}

        sdist_path = sdist_files[0]
        issues = []

        try:
            with tarfile.open(sdist_path, "r:gz") as tar:
                files = tar.getnames()

                # Check for required files
                required_files = ["pyproject.toml", "README.md", "LICENSE"]

                for req_file in required_files:
                    if not any(f.endswith(req_file) for f in files):
                        issues.append(f"Missing required file: {req_file}")

                # Check package structure
                package_files = [
                    f for f in files if "/pulse/" in f and f.endswith(".py")
                ]
                if not package_files:
                    issues.append("No Python package files found in sdist")

                # Check for unwanted files
                unwanted_patterns = [
                    "__pycache__",
                    ".pyc",
                    ".git",
                    "node_modules",
                    ".DS_Store",
                ]

                for pattern in unwanted_patterns:
                    unwanted_files = [f for f in files if pattern in f]
                    if unwanted_files:
                        issues.append(
                            f"Unwanted files found: {pattern} "
                            f"({len(unwanted_files)} files)"
                        )

        except Exception as e:
            return {"success": False, "error": f"Failed to validate sdist: {e}"}

        return {
            "success": len(issues) == 0,
            "issues": issues,
            "sdist_file": str(sdist_path.name),
        }

    def check_twine_validation(self) -> Dict[str, Any]:
        """Run twine check on built packages."""
        print("🔍 Running twine validation...")

        # Check if twine is available
        result = self.run_command([sys.executable, "-m", "twine", "--version"])
        if not result["success"]:
            return {"success": False, "error": "twine not available"}

        # Run twine check
        result = self.run_command(
            [sys.executable, "-m", "twine", "check", str(self.dist_dir / "*")]
        )

        return {
            "success": result["success"],
            "output": result["stdout"],
            "errors": result["stderr"],
        }

    def validate_installation(self) -> Dict[str, Any]:
        """Test installation in clean environment."""
        print("🧪 Testing installation in clean environment...")

        wheel_files = list(self.dist_dir.glob("*.whl"))
        if not wheel_files:
            return {"success": False, "error": "No wheel file to test"}

        wheel_path = wheel_files[0]

        # Create temporary virtual environment
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "test_venv"

            # Create venv
            result = self.run_command([sys.executable, "-m", "venv", str(venv_dir)])
            if not result["success"]:
                return {"success": False, "error": "Failed to create test venv"}

            # Determine python command
            if os.name == "nt":  # Windows
                python_cmd = str(venv_dir / "Scripts" / "python.exe")
            else:  # Unix-like
                python_cmd = str(venv_dir / "bin" / "python")

            # Install wheel
            result = self.run_command(
                [python_cmd, "-m", "pip", "install", str(wheel_path)]
            )
            if not result["success"]:
                return {
                    "success": False,
                    "error": "Failed to install wheel",
                    "details": result,
                }

            # Test import
            test_script = """
import sys
try:
    import pulse
    from pulse.core.client import CoreClient
    print(f"✅ Successfully imported Pulse SDK {pulse.__version__}")
    print(f"Python: {sys.version}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
"""

            result = self.run_command([python_cmd, "-c", test_script])

            return {
                "success": result["success"],
                "output": result["stdout"],
                "errors": result["stderr"],
            }

    def run_all_validations(self):
        """Run all package validations."""
        print("🚀 Starting Pulse SDK Package Validation")
        print("=" * 50)

        validations = [
            ("pyproject.toml", self.validate_pyproject_toml),
            ("build", self.build_package),
            ("wheel", self.validate_wheel),
            ("sdist", self.validate_sdist),
            ("twine", self.check_twine_validation),
            ("installation", self.validate_installation),
        ]

        for name, validation_func in validations:
            try:
                result = validation_func()
                self.results[name] = result

                if result["success"]:
                    print(f"✅ {name}: PASSED")
                else:
                    print(f"❌ {name}: FAILED")
                    if "error" in result:
                        print(f"   Error: {result['error']}")
                    if "issues" in result and result["issues"]:
                        for issue in result["issues"][:3]:  # Show first 3 issues
                            print(f"   - {issue}")
                        if len(result["issues"]) > 3:
                            print(f"   ... and {len(result['issues']) - 3} more issues")

            except Exception as e:
                print(f"❌ {name}: EXCEPTION - {str(e)}")
                self.results[name] = {"success": False, "error": str(e)}

        self.print_summary()

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)

        total_validations = len(self.results)
        passed_validations = sum(1 for r in self.results.values() if r["success"])

        print(f"Total validations: {total_validations}")
        print(f"Passed: {passed_validations}")
        print(f"Failed: {total_validations - passed_validations}")
        print(f"Success rate: {passed_validations/total_validations*100:.1f}%")

        print("\nDetailed Results:")
        for name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {name:15} {status}")

        if passed_validations == total_validations:
            print("\n🎉 All package validations passed!")
            print("   Your package follows Python packaging best practices.")
        else:
            print(
                f"\n⚠️  {total_validations - passed_validations} validation(s) failed."
            )
            print("   Review the issues above before publishing.")


def main():
    """Main validation function."""
    validator = PackageValidator()
    validator.run_all_validations()

    # Exit with error code if any validations failed
    failed_validations = sum(1 for r in validator.results.values() if not r["success"])
    sys.exit(failed_validations)


if __name__ == "__main__":
    main()
