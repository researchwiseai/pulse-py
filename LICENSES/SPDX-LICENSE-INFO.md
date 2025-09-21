# SPDX License Information

## Overview

This document provides comprehensive SPDX (Software Package Data Exchange) license information for the Pulse SDK and all its dependencies. SPDX is an open standard for communicating software bill of material information, including components, licenses, copyrights, and security references.

## Project License Information

### Main Package
- **Package Name**: pulse-sdk
- **SPDX License ID**: Apache-2.0
- **License File**: [LICENSE](../LICENSE)
- **Copyright**: Copyright (c) 2025 Researchwise AI
- **Homepage**: https://github.com/researchwiseai/pulse-sdk
- **Download Location**: https://pypi.org/project/pulse-sdk/

### SPDX Document Identifier
```
SPDXVersion: SPDX-2.3
DataLicense: CC0-1.0
SPDXID: SPDXRef-DOCUMENT
DocumentName: Pulse SDK SPDX Document
DocumentNamespace: https://github.com/researchwise/pulse-sdk/spdx-2024-12-19
Creator: Tool: pulse-sdk-license-checker
Created: 2024-12-19T00:00:00Z
```

## License Compatibility Matrix

The Pulse SDK uses the Apache-2.0 license, which is compatible with the following licenses:

### Compatible Licenses (✅)
- **Apache-2.0**: Fully compatible
- **MIT**: Fully compatible
- **BSD-2-Clause**: Fully compatible
- **BSD-3-Clause**: Fully compatible
- **ISC**: Fully compatible
- **Unlicense**: Fully compatible

### Conditionally Compatible (⚠️)
- **LGPL-2.1**: Compatible for dynamic linking
- **LGPL-3.0**: Compatible for dynamic linking
- **MPL-2.0**: Compatible with proper attribution
- **GPL-3.0**: Compatible (Apache-2.0 can be combined with GPL-3.0+)

### Incompatible Licenses (❌)
- **GPL-2.0**: Incompatible (Apache-2.0 cannot be combined with GPL-2.0)
- **AGPL-3.0**: Generally incompatible (network copyleft conflicts)

## Dependency License Analysis

### Core Dependencies

#### Production Dependencies
```spdx
PackageName: httpx
SPDXID: SPDXRef-Package-httpx
PackageVersion: 0.25.2
PackageDownloadLocation: https://pypi.org/project/httpx/
FilesAnalyzed: false
PackageLicenseConcluded: BSD-3-Clause
PackageLicenseDeclared: BSD-3-Clause
PackageCopyrightText: Copyright (c) 2019, Encode OSS Ltd.
```

```spdx
PackageName: pydantic
SPDXID: SPDXRef-Package-pydantic
PackageVersion: 2.5.0
PackageDownloadLocation: https://pypi.org/project/pydantic/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2017 to present Pydantic Services Inc.
```

```spdx
PackageName: diskcache
SPDXID: SPDXRef-Package-diskcache
PackageVersion: 5.6.3
PackageDownloadLocation: https://pypi.org/project/diskcache/
FilesAnalyzed: false
PackageLicenseConcluded: Apache-2.0
PackageLicenseDeclared: Apache-2.0
PackageCopyrightText: Copyright 2016-2023 Grant Jenks
```

#### Development Dependencies
```spdx
PackageName: pytest
SPDXID: SPDXRef-Package-pytest
PackageVersion: 7.4.3
PackageDownloadLocation: https://pypi.org/project/pytest/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2004 Holger Krekel and others
```

```spdx
PackageName: black
SPDXID: SPDXRef-Package-black
PackageVersion: 23.11.0
PackageDownloadLocation: https://pypi.org/project/black/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2018 Łukasz Langa
```

### License Risk Assessment

#### Low Risk Dependencies (MIT/BSD/Apache-2.0)
- ✅ **httpx**: BSD-3-Clause - Permissive license, commercial use allowed
- ✅ **pydantic**: MIT - Fully compatible with project license
- ✅ **diskcache**: Apache-2.0 - Permissive license with patent grant
- ✅ **pytest**: MIT - Development dependency, fully compatible
- ✅ **black**: MIT - Development dependency, fully compatible

#### Medium Risk Dependencies (LGPL/MPL)
- ⚠️ **None currently identified** - All dependencies use permissive licenses

#### High Risk Dependencies (GPL/AGPL)
- ❌ **None currently identified** - Project policy prohibits GPL dependencies

## License Compliance Procedures

### Automated License Checking

The project includes automated license checking through the `scripts/license_checker.py` tool:

```bash
# Check license compatibility
python scripts/license_checker.py --format text

# Generate SPDX document
python scripts/license_checker.py --format spdx --output SPDX-LICENSE.json

# Fail CI on license issues
python scripts/license_checker.py --fail-on-issues
```

### Manual Review Process

1. **New Dependency Addition**:
   - Check license compatibility using automated tools
   - Review license text for any unusual terms
   - Document decision in dependency review log
   - Update SPDX documentation

2. **Dependency Updates**:
   - Verify license hasn't changed
   - Check for new license requirements
   - Update compatibility matrix if needed

3. **License Change Detection**:
   - Automated monitoring in CI/CD pipeline
   - Alert on any license changes
   - Manual review required for approval

### Compliance Documentation

#### Required Attribution Files
- `LICENSE`: Main project license (MIT)
- `LICENSES/MIT.txt`: Full MIT license text
- `LICENSES/SPDX-LICENSE-INFO.md`: This document
- `THIRD-PARTY-LICENSES.md`: Third-party license attributions

#### SPDX Document Generation
```python
# Generate complete SPDX document
from scripts.license_checker import LicenseChecker

checker = LicenseChecker()
analysis = checker.analyze_dependencies()
spdx_doc = checker.generate_spdx_document(analysis)

# Save SPDX document
with open('SPDX-LICENSE.json', 'w') as f:
    json.dump(spdx_doc, f, indent=2)
```

## Third-Party License Attributions

### MIT Licensed Components
```
Copyright (c) 2017 to present Pydantic Services Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy...
[Full MIT license text]
```

### BSD Licensed Components
```
Copyright (c) 2019, Encode OSS Ltd.
Redistribution and use in source and binary forms, with or without modification...
[Full BSD license text]
```

### Apache Licensed Components
```
Copyright 2016-2023 Grant Jenks
Licensed under the Apache License, Version 2.0 (the "License")...
[Full Apache license text]
```

## Enterprise License Considerations

### Commercial Use
- ✅ **Permitted**: All dependencies allow commercial use
- ✅ **No Royalties**: No royalty requirements for any dependencies
- ✅ **Distribution**: Can be distributed in commercial products

### Copyleft Obligations
- ✅ **None**: No copyleft obligations from current dependencies
- ✅ **Source Code**: No requirement to provide source code
- ✅ **Modifications**: No requirement to share modifications

### Patent Considerations
- ✅ **Apache-2.0**: Includes explicit patent grant
- ✅ **MIT/BSD**: No patent issues identified
- ✅ **Overall**: No patent licensing concerns

## Compliance Monitoring

### Automated Checks
```yaml
# GitHub Actions workflow for license compliance
name: License Compliance Check
on: [push, pull_request]

jobs:
  license-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Check license compliance
        run: python scripts/license_checker.py --fail-on-issues
      - name: Generate SPDX document
        run: python scripts/license_checker.py --format spdx --output spdx-report.json
      - name: Upload SPDX document
        uses: actions/upload-artifact@v3
        with:
          name: spdx-document
          path: spdx-report.json
```

### Manual Review Schedule
- **Weekly**: Review new dependencies added
- **Monthly**: Full license compatibility audit
- **Quarterly**: Update SPDX documentation
- **Annually**: Comprehensive license review

## Contact Information

### License Compliance Team
- **Email**: [legal@researchwise.ai](mailto:legal@researchwise.ai)
- **Role**: License compliance and legal review
- **Response Time**: 2-3 business days

### Security Team
- **Email**: [security@researchwise.ai](mailto:security@researchwise.ai)
- **Role**: Security implications of license changes
- **Response Time**: 1-2 business days

## References

### Standards and Specifications
- [SPDX Specification 2.3](https://spdx.github.io/spdx-spec/)
- [SPDX License List](https://spdx.org/licenses/)
- [OpenChain Specification](https://www.openchainproject.org/)

### Tools and Resources
- [SPDX Tools](https://github.com/spdx/tools-python)
- [License Compatibility Matrix](https://www.gnu.org/licenses/license-list.html)
- [OSI Approved Licenses](https://opensource.org/licenses/)

### Legal Resources
- [Software Freedom Law Center](https://www.softwarefreedom.org/)
- [Free Software Foundation](https://www.fsf.org/)
- [Open Source Initiative](https://opensource.org/)

---

**Document Version**: 1.0
**Last Updated**: December 2024
**Next Review**: March 2025
**SPDX Version**: 2.3
**License List Version**: 3.21
