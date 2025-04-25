#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {alpha|beta|patch|feature|breaking}" >&2
  exit 1
}

if [ $# -ne 1 ]; then
  usage
fi
mode=$1

# Extract current version from pyproject.toml
current_version=$(grep -E '^version = "' pyproject.toml | head -1 | sed -E 's/version = "([^"]+)"/\1/')
if [[ -z "$current_version" ]]; then
  echo "Error: Couldn't find current version in pyproject.toml" >&2
  exit 1
fi

# Parse version components
ver_regex='^([0-9]+)\.([0-9]+)\.([0-9]+)(a|b)?([0-9]*)$'
if [[ "$current_version" =~ $ver_regex ]]; then
  major=${BASH_REMATCH[1]}
  minor=${BASH_REMATCH[2]}
  patch=${BASH_REMATCH[3]}
  pre_tag=${BASH_REMATCH[4]}
  pre_num=${BASH_REMATCH[5]}
else
  echo "Error: Version '$current_version' is not in X.Y.Z[aN] or X.Y.Z[bN] format" >&2
  exit 1
fi

case "$mode" in
  patch)
    patch=$((patch + 1)); new_version="${major}.${minor}.${patch}";;
  feature)
    minor=$((minor + 1)); patch=0; new_version="${major}.${minor}.0";;
  breaking)
    major=$((major + 1)); minor=0; patch=0; new_version="${major}.0.0";;
  alpha)
    if [[ "$pre_tag" == "a" ]]; then
      pre_num=$((pre_num + 1))
    else
      pre_num=1
    fi
    new_version="${major}.${minor}.${patch}a${pre_num}";;
  beta)
    if [[ "$pre_tag" == "b" ]]; then
      pre_num=$((pre_num + 1))
    else
      pre_num=1
    fi
    new_version="${major}.${minor}.${patch}b${pre_num}";;
  *)
    usage
    ;;
esac

echo "Bumping version: $current_version -> $new_version"

# Update pyproject.toml
sed -i -E "s#^version = \".*\"#version = \"$new_version\"#" pyproject.toml

# Update package __version__
sed -i -E "s#__version__ = \".*\"#__version__ = \"$new_version\"#" pulse_client/__init__.py

# Commit and tag
git add pyproject.toml pulse_client/__init__.py
git commit -m "chore: bump version to $new_version"
tag="v$new_version"
git tag "$tag"

echo
echo "Created commit and tag $tag. To push, run:" \
     && echo "  git push origin main && git push origin $tag"
echo
exit 0
