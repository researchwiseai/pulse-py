# Third-Party Attributions

This file contains the licenses and notices for third-party software
included in or used by the Pulse SDK.

This document is maintained in the repository root and linked from release notes
to provide compliance information without duplicating files as release assets.

## Direct Dependencies

### httpx
- **License**: BSD 3-Clause License
- **Copyright**: Copyright (c) 2019, Encode OSS Ltd.
- **Homepage**: https://github.com/encode/httpx
- **License Text**: See LICENSES/BSD-3-Clause-httpx.txt

### pydantic
- **License**: MIT License
- **Copyright**: Copyright (c) 2017 to present Pydantic Services Inc.
- **Homepage**: https://github.com/pydantic/pydantic
- **License Text**: See LICENSES/MIT-pydantic.txt

### diskcache
- **License**: Apache License 2.0
- **Copyright**: Copyright 2016-2023 Grant Jenks
- **Homepage**: https://github.com/grantjenks/python-diskcache
- **License Text**: See LICENSES/Apache-2.0-diskcache.txt

## Development Dependencies

### pytest
- **License**: MIT License
- **Copyright**: Copyright (c) 2004 Holger Krekel and others
- **Homepage**: https://github.com/pytest-dev/pytest
- **Usage**: Testing framework (development only)

### black
- **License**: MIT License
- **Copyright**: Copyright (c) 2018 Łukasz Langa
- **Homepage**: https://github.com/psf/black
- **Usage**: Code formatting (development only)

## License Compatibility

All third-party licenses are compatible with the Apache License 2.0:

- **MIT License**: Fully compatible, can be combined
- **BSD 3-Clause**: Fully compatible, can be combined
- **Apache 2.0**: Same license, fully compatible

## Verification

This attribution file was generated automatically. To verify the accuracy:

```bash
# Check current dependencies
pip list --format=json > current-deps.json

# Scan licenses
pip-licenses --format=json --output-file=license-scan.json

# Compare with this file
python scripts/verify_attributions.py
```

---
Generated: 2025-09-21T15:32:12.429636+00:00
Generator: pulse-sdk-compliance-generator v1.0
