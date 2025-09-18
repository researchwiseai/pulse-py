# Changelog

## [0.4.0](https://github.com/researchwiseai/pulse-py/compare/pulse-sdk-v0.3.3...pulse-sdk-v0.4.0) (2025-09-18)


### Features

* **analyzer:** add persistent on-disk caching with clear_cache and context-manager; chore: refine .gitignore ([f077956](https://github.com/researchwiseai/pulse-py/commit/f077956d4f454ea07697d3efa2ca9aeb0ee2839b))
* enhance supply chain security with SLSA attestations and SBOM generation ([423f15c](https://github.com/researchwiseai/pulse-py/commit/423f15cbebf78e5d0c141fd881005c925af2ea5d))
* support advanced extraction options ([#16](https://github.com/researchwiseai/pulse-py/issues/16)) ([f494a9e](https://github.com/researchwiseai/pulse-py/commit/f494a9e6e1f445230eb36b7edcf9199b95758fbe))
* update extraction api ([fcad2f9](https://github.com/researchwiseai/pulse-py/commit/fcad2f9591299a0b57d2aaeec739d1c74c0596ca))


### Bug Fixes

* Add relative_files=true to coverage configuration ([6efa93c](https://github.com/researchwiseai/pulse-py/commit/6efa93c907095a324df07aaf15e09b05039f7859))
* configure release-please to use PAT token for PR creation ([75912b9](https://github.com/researchwiseai/pulse-py/commit/75912b9e81cad3242fae9e724cd0e1cb47bd6b1c))
* correct .gitignore patterns to ignore __pycache__/ and *.pyc ([6ea8b23](https://github.com/researchwiseai/pulse-py/commit/6ea8b23c2d4c0237bfa45e92e7145de7b7c975f1))
* Decoupled base url and audience ([d04961f](https://github.com/researchwiseai/pulse-py/commit/d04961fad99e058d36b22b375bc0e15b59122c40))
* Fixed bug in Py project TOML ([61ee026](https://github.com/researchwiseai/pulse-py/commit/61ee026a21098422385d3ea352a00ddd591c662d))
* Minor fix to test suite and version bump ([4d0ec72](https://github.com/researchwiseai/pulse-py/commit/4d0ec72f8315208b893bb736cc2b1b118c549a04))
* Removed warnings from test suite ([5d1efba](https://github.com/researchwiseai/pulse-py/commit/5d1efbad28383d26a81f07e897f7d498acc52633))
* resolve all GitHub Actions workflow failures ([1af0403](https://github.com/researchwiseai/pulse-py/commit/1af040396cea8c0ec4a915018ff3bb4721d9eef0))
* resolve PKCE auth test failure by properly handling empty environment variables ([c2a0b41](https://github.com/researchwiseai/pulse-py/commit/c2a0b4180afbd86feebbd436ca3d999b15d3933a))


### Documentation

* update README and API docs ([#13](https://github.com/researchwiseai/pulse-py/issues/13)) ([4e24d80](https://github.com/researchwiseai/pulse-py/commit/4e24d800ef407ba765fa37a1eb2012eead6f6f59))


### Performance Improvements

* optimize fmt target to only format source directories ([a908f24](https://github.com/researchwiseai/pulse-py/commit/a908f243dd63b7fdc925691e30cf061dbab47074))

## [0.3.3](https://github.com/researchwiseai/pulse-py/compare/v0.3.2...v0.3.3) (2024-12-XX)

### Bug Fixes

* decouple base URL and OAuth audience configuration to avoid unintended coupling between environments

## [0.3.2](https://github.com/researchwiseai/pulse-py/compare/v0.3.1...v0.3.2) (2024-12-XX)

### Bug Fixes

* improve 401 Unauthorized diagnostics: PulseAPIError now includes AWS API Gateway hints when available (e.g., `www-authenticate`, `x-amzn-errortype`, `apigw-requestid`). This makes it easier to troubleshoot token and audience issues.

---

*Note: Starting with the next release, this changelog will be automatically generated using [Release Please](https://github.com/googleapis/release-please) based on [Conventional Commits](https://www.conventionalcommits.org/).*
