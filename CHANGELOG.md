# Changelog

## [0.3.3](https://github.com/researchwiseai/pulse-py/compare/v0.3.2...v0.3.3) (2024-12-XX)

### Bug Fixes

* decouple base URL and OAuth audience configuration to avoid unintended coupling between environments

## [0.3.2](https://github.com/researchwiseai/pulse-py/compare/v0.3.1...v0.3.2) (2024-12-XX)

### Bug Fixes

* improve 401 Unauthorized diagnostics: PulseAPIError now includes AWS API Gateway hints when available (e.g., `www-authenticate`, `x-amzn-errortype`, `apigw-requestid`). This makes it easier to troubleshoot token and audience issues.

---

*Note: Starting with the next release, this changelog will be automatically generated using [Release Please](https://github.com/googleapis/release-please) based on [Conventional Commits](https://www.conventionalcommits.org/).*
