"""
Centralized security configuration and utilities for CI/CD workflows.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SecurityThresholds:
    """Security scan thresholds configuration."""

    bandit_min_severity: str = "medium"
    bandit_min_confidence: str = "medium"
    bandit_fail_on_high: bool = True
    bandit_fail_on_medium: bool = True

    pip_audit_fail_on_critical: bool = True
    pip_audit_fail_on_high: bool = True

    # Coverage thresholds
    min_coverage_percent: float = 90.0


@dataclass
class SecurityResults:
    """Security scan results."""

    bandit_high_issues: int = 0
    bandit_medium_issues: int = 0
    bandit_total_issues: int = 0

    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    total_vulnerabilities: int = 0

    coverage_percent: Optional[float] = None


class SecurityAnalyzer:
    """Analyze security scan results and determine pass/fail status."""

    def __init__(self, thresholds: SecurityThresholds):
        self.thresholds = thresholds

    def analyze_bandit_results(
        self, report_path: str = "bandit-report.json"
    ) -> Tuple[int, int, int]:
        """Analyze Bandit JSON report and return (high, medium, total) issue counts."""
        if not os.path.exists(report_path):
            return 0, 0, 0

        with open(report_path) as f:
            data = json.load(f)

        results = data.get("results", [])
        high_issues = sum(1 for r in results if r.get("issue_severity") == "HIGH")
        medium_issues = sum(1 for r in results if r.get("issue_severity") == "MEDIUM")
        total_issues = len(results)

        return high_issues, medium_issues, total_issues

    def analyze_pip_audit_results(
        self, report_path: str = "pip-audit-report.json"
    ) -> Tuple[int, int, int]:
        """Analyze pip-audit JSON report and return vulnerability counts."""
        if not os.path.exists(report_path):
            return 0, 0, 0

        with open(report_path) as f:
            data = json.load(f)

        critical_vulns = 0
        high_vulns = 0
        total_vulns = len(data.get("vulnerabilities", []))

        for vuln in data.get("vulnerabilities", []):
            for alias in vuln.get("aliases", []):
                severity = alias.get("severity", "").upper()
                if severity == "CRITICAL":
                    critical_vulns += 1
                elif severity == "HIGH":
                    high_vulns += 1

        return critical_vulns, high_vulns, total_vulns

    def analyze_coverage_results(
        self, report_path: str = "coverage.json"
    ) -> Optional[float]:
        """Analyze coverage JSON report and return coverage percentage."""
        if not os.path.exists(report_path):
            return None

        with open(report_path) as f:
            data = json.load(f)

        return data.get("totals", {}).get("percent_covered", 0.0)

    def analyze_all(self) -> SecurityResults:
        """Analyze all security scan results."""
        bandit_high, bandit_medium, bandit_total = self.analyze_bandit_results()
        critical_vulns, high_vulns, total_vulns = self.analyze_pip_audit_results()
        coverage = self.analyze_coverage_results()

        return SecurityResults(
            bandit_high_issues=bandit_high,
            bandit_medium_issues=bandit_medium,
            bandit_total_issues=bandit_total,
            critical_vulnerabilities=critical_vulns,
            high_vulnerabilities=high_vulns,
            total_vulnerabilities=total_vulns,
            coverage_percent=coverage,
        )

    def check_security_gates(self, results: SecurityResults) -> Tuple[bool, List[str]]:
        """Check if all security gates pass. Returns (passed, failure_reasons)."""
        failures = []

        # Check Bandit results
        if self.thresholds.bandit_fail_on_high and results.bandit_high_issues > 0:
            failures.append(
                f"Found {results.bandit_high_issues} high severity Bandit issues"
            )

        if self.thresholds.bandit_fail_on_medium and results.bandit_medium_issues > 0:
            failures.append(
                f"Found {results.bandit_medium_issues} medium severity Bandit issues"
            )

        # Check vulnerability results
        if (
            self.thresholds.pip_audit_fail_on_critical
            and results.critical_vulnerabilities > 0
        ):
            failures.append(
                f"Found {results.critical_vulnerabilities} critical vulnerabilities"
            )

        if self.thresholds.pip_audit_fail_on_high and results.high_vulnerabilities > 0:
            failures.append(
                f"Found {results.high_vulnerabilities} high severity vulnerabilities"
            )

        # Check coverage
        if (
            results.coverage_percent is not None
            and results.coverage_percent < self.thresholds.min_coverage_percent
        ):
            failures.append(
                f"Coverage {results.coverage_percent:.1f}% below "
                f"threshold {self.thresholds.min_coverage_percent}%"
            )

        return len(failures) == 0, failures

    def generate_summary(self, results: SecurityResults) -> str:
        """Generate a human-readable security summary."""
        passed, failures = self.check_security_gates(results)

        summary = ["🔒 Security Scan Summary", ""]

        # Bandit results
        summary.extend(
            [
                "## 🛡️ SAST Scan (Bandit)",
                f"- **Total Issues**: {results.bandit_total_issues}",
                f"- **High Severity**: {results.bandit_high_issues}",
                f"- **Medium Severity**: {results.bandit_medium_issues}",
                "",
            ]
        )

        # Vulnerability results
        summary.extend(
            [
                "## 🔍 Dependency Scan (pip-audit)",
                f"- **Total Vulnerabilities**: {results.total_vulnerabilities}",
                f"- **Critical**: {results.critical_vulnerabilities}",
                f"- **High**: {results.high_vulnerabilities}",
                "",
            ]
        )

        # Coverage results
        if results.coverage_percent is not None:
            summary.extend(
                [
                    "## 📊 Test Coverage",
                    f"- **Coverage**: {results.coverage_percent:.1f}%",
                    f"- **Threshold**: {self.thresholds.min_coverage_percent}%",
                    "",
                ]
            )

        # Overall status
        if passed:
            summary.append("## ✅ All Security Gates Passed")
        else:
            summary.extend(["## ❌ Security Gate Failures", ""])
            for failure in failures:
                summary.append(f"- {failure}")

        return "\n".join(summary)


def main():
    """CLI entry point for security analysis."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze security scan results")
    parser.add_argument(
        "--bandit-report",
        default="bandit-report.json",
        help="Path to Bandit JSON report",
    )
    parser.add_argument(
        "--pip-audit-report",
        default="pip-audit-report.json",
        help="Path to pip-audit JSON report",
    )
    parser.add_argument(
        "--coverage-report",
        default="coverage.json",
        help="Path to coverage JSON report",
    )
    parser.add_argument(
        "--fail-on-medium",
        action="store_true",
        help="Fail on medium severity Bandit issues",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't exit with error",
    )

    args = parser.parse_args()

    # Configure thresholds
    thresholds = SecurityThresholds(bandit_fail_on_medium=args.fail_on_medium)

    analyzer = SecurityAnalyzer(thresholds)
    results = analyzer.analyze_all()

    # Print summary
    print(analyzer.generate_summary(results))

    # Check gates and exit appropriately
    passed, failures = analyzer.check_security_gates(results)

    if not passed and not args.summary_only:
        print("\n❌ Security checks failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    elif passed:
        print("\n✅ All security checks passed")


if __name__ == "__main__":
    main()
