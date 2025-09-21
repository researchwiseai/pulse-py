# Changelog

## [0.4.1](https://github.com/researchwiseai/pulse-py/compare/pulse-sdk-v0.4.0...pulse-sdk-v0.4.1) (2025-09-21)


### Bug Fixes

* separate signature files from dist directory in publish workflow ([1f34319](https://github.com/researchwiseai/pulse-py/commit/1f34319282ed6928d48a79f76b7ad4fa19a3b41e))

## [0.4.0](https://github.com/researchwiseai/pulse-py/compare/pulse-sdk-v0.3.3...pulse-sdk-v0.4.0) (2025-09-21)


### Features

* Add comprehensive error recovery documentation ([a350d43](https://github.com/researchwiseai/pulse-py/commit/a350d4357ee6e0a1f1f29ca81261d973506c7881))
* **analyzer:** add persistent on-disk caching with clear_cache and context-manager; chore: refine .gitignore ([f077956](https://github.com/researchwiseai/pulse-py/commit/f077956d4f454ea07697d3efa2ca9aeb0ee2839b))
* create comprehensive quick start documentation ([92869e6](https://github.com/researchwiseai/pulse-py/commit/92869e62c1faad8cca11a9bb8fce032811b33a79))
* enhance CI/CD pipeline with comprehensive security and quality checks ([b50982b](https://github.com/researchwiseai/pulse-py/commit/b50982bba32e677f1f6d65a2e06d0f4f76ce5966))
* enhance supply chain security with SLSA attestations and SBOM generation ([423f15c](https://github.com/researchwiseai/pulse-py/commit/423f15cbebf78e5d0c141fd881005c925af2ea5d))
* implement automated documentation testing ([60e242b](https://github.com/researchwiseai/pulse-py/commit/60e242baa3a6a50bdc01df81941daaea23b277e8))
* implement comprehensive debugging and introspection tools ([57e5230](https://github.com/researchwiseai/pulse-py/commit/57e52302cd117a9a04807417523a6314379008de))
* implement comprehensive input validation and enhanced error handling ([df5a501](https://github.com/researchwiseai/pulse-py/commit/df5a5013e2bdad86660eb836a1b175aa31e51241))
* remove dev/staging environment references ([4cf6a4e](https://github.com/researchwiseai/pulse-py/commit/4cf6a4e0a9f4f7de313c4001fbbf4e71188755a8))
* simplify installation and dependency management ([75b39d9](https://github.com/researchwiseai/pulse-py/commit/75b39d9e95f2db072697e7bd4093da54adada950))
* support advanced extraction options ([#16](https://github.com/researchwiseai/pulse-py/issues/16)) ([f494a9e](https://github.com/researchwiseai/pulse-py/commit/f494a9e6e1f445230eb36b7edcf9199b95758fbe))
* update core Pydantic models for OpenAPI v0.9.0 support ([6d57d61](https://github.com/researchwiseai/pulse-py/commit/6d57d6130222ed77488caa7d1214ae9af2a946c7))
* update CoreClient methods to support OpenAPI v0.9.0 parameters ([5a1232c](https://github.com/researchwiseai/pulse-py/commit/5a1232c4200af3684de1fd6c01e4e5198a967495))
* update extraction api ([fcad2f9](https://github.com/researchwiseai/pulse-py/commit/fcad2f9591299a0b57d2aaeec739d1c74c0596ca))
* update higher-level API components for OpenAPI v0.9.0 ([072024a](https://github.com/researchwiseai/pulse-py/commit/072024ae74021a04844c5dd9416b137f59524cfa))


### Bug Fixes

* Add dev dependencies to CI workflow to include black formatter ([a39f74a](https://github.com/researchwiseai/pulse-py/commit/a39f74affc74f0b95ce628a1d393ef4ff6fb7770))
* Add dev dependencies to CI workflow to include black formatter ([34e4fb8](https://github.com/researchwiseai/pulse-py/commit/34e4fb85a7971a6432498430cdb622e45a755860))
* Add relative_files=true to coverage configuration ([6efa93c](https://github.com/researchwiseai/pulse-py/commit/6efa93c907095a324df07aaf15e09b05039f7859))
* configure release-please to use PAT token for PR creation ([75912b9](https://github.com/researchwiseai/pulse-py/commit/75912b9e81cad3242fae9e724cd0e1cb47bd6b1c))
* correct .gitignore patterns to ignore __pycache__/ and *.pyc ([6ea8b23](https://github.com/researchwiseai/pulse-py/commit/6ea8b23c2d4c0237bfa45e92e7145de7b7c975f1))
* correct signature verification to only check distribution files ([90a51b5](https://github.com/researchwiseai/pulse-py/commit/90a51b5e8004a96f81f09c36f8776198786e7aa4))
* Decoupled base url and audience ([d04961f](https://github.com/researchwiseai/pulse-py/commit/d04961fad99e058d36b22b375bc0e15b59122c40))
* Fixed bug in Py project TOML ([61ee026](https://github.com/researchwiseai/pulse-py/commit/61ee026a21098422385d3ea352a00ddd591c662d))
* Getting CI green for GA ([eac5d85](https://github.com/researchwiseai/pulse-py/commit/eac5d85f6676d612423fcdd83738f71d93b0f340))
* Minor fix to test suite and version bump ([4d0ec72](https://github.com/researchwiseai/pulse-py/commit/4d0ec72f8315208b893bb736cc2b1b118c549a04))
* Removed warnings from test suite ([5d1efba](https://github.com/researchwiseai/pulse-py/commit/5d1efbad28383d26a81f07e897f7d498acc52633))
* resolve all GitHub Actions workflow failures ([1af0403](https://github.com/researchwiseai/pulse-py/commit/1af040396cea8c0ec4a915018ff3bb4721d9eef0))
* resolve artifact naming conflicts and update coverage threshold ([b00976d](https://github.com/researchwiseai/pulse-py/commit/b00976db8b310d2384da9bc7957e7657f97af987))
* resolve PKCE auth test failure by properly handling empty environment variables ([c2a0b41](https://github.com/researchwiseai/pulse-py/commit/c2a0b4180afbd86feebbd436ca3d999b15d3933a))
* resolve security issues identified by Bandit scan ([b927c8a](https://github.com/researchwiseai/pulse-py/commit/b927c8a331e851cfc4c99bce5427bcbda4920d39))
* resolve security issues identified by Bandit scan ([6facc31](https://github.com/researchwiseai/pulse-py/commit/6facc31367149e2b53f6b9f833f99997adff360c))
* update tests for v0.9.0 API changes ([007d24a](https://github.com/researchwiseai/pulse-py/commit/007d24a939cb8bbec358d6e2e9be2ad111542384))


### Documentation

* update documentation and examples for OpenAPI v0.9.0 ([42808ba](https://github.com/researchwiseai/pulse-py/commit/42808baa0bbaf9617bcd5543a5dc96dd87e0e02f))
* update README and API docs ([#13](https://github.com/researchwiseai/pulse-py/issues/13)) ([4e24d80](https://github.com/researchwiseai/pulse-py/commit/4e24d800ef407ba765fa37a1eb2012eead6f6f59))


### Code Refactoring

* consolidate security and compliance checks ([367a82e](https://github.com/researchwiseai/pulse-py/commit/367a82e5d05cb0526b7241a52d27940ba7b1709c))
* eliminate duplicate security scans and optimize workflows ([356a33a](https://github.com/researchwiseai/pulse-py/commit/356a33aedb19e115a358a542d8f4cdac93534263))


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
