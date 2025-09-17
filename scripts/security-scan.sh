#!/bin/bash

# Security scanning script for Pulse SDK
# This script runs the same security scans as the CI pipeline

set -e

echo "🔒 Running security scans for Pulse SDK..."
echo

# Create reports directory
mkdir -p security-reports

# Run Bandit SAST scan
echo "📊 Running Bandit SAST scan..."
python3 -m bandit -r pulse \
  --exclude pulse/core/.ipynb_checkpoints \
  --skip B101,B110,B105,B311,B403,B601 \
  -f json -o security-reports/bandit-report.json

python3 -m bandit -r pulse \
  --exclude pulse/core/.ipynb_checkpoints \
  --skip B101,B110,B105,B311,B403,B601 \
  -f txt -o security-reports/bandit-report.txt

echo "✅ Bandit scan completed"
echo

# Run pip-audit vulnerability scan
echo "🔍 Running pip-audit vulnerability scan..."
python3 -m pip_audit --format=json --output=security-reports/pip-audit-report.json
python3 -m pip_audit --format=columns

echo "✅ pip-audit scan completed"
echo

echo "🎉 Security scans completed successfully!"
echo "📁 Reports saved in security-reports/ directory:"
echo "  - bandit-report.json"
echo "  - bandit-report.txt"
echo "  - pip-audit-report.json"