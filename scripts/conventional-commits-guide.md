# Conventional Commits Guide

This project uses [Conventional Commits](https://www.conventionalcommits.org/) to automatically generate changelogs and manage releases.

## Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

## Examples

```bash
# Feature
git commit -m "feat: add sentiment analysis caching"

# Bug fix
git commit -m "fix: handle network timeout in auth flow"

# Documentation
git commit -m "docs: update quick start guide"

# Breaking change
git commit -m "feat!: change API response format"
# or
git commit -m "feat: change API response format

BREAKING CHANGE: The response format has changed from array to object"
```

## Breaking Changes

- Add `!` after the type/scope: `feat!: breaking change`
- Or add `BREAKING CHANGE:` in the footer

## Scopes (Optional)

You can add a scope to provide additional context:

- `feat(auth): add PKCE flow support`
- `fix(core): resolve timeout handling`
- `docs(api): update client examples`

## Validation

- Pre-commit hooks validate commit messages locally
- CI validates PR titles follow conventional commit format
- Release Please uses these commits to generate changelogs automatically
