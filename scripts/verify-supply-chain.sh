#!/bin/bash

# Supply Chain Security Verification Script
# This script helps verify the integrity and authenticity of Pulse SDK releases

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO="researchwiseai/pulse-py"
COSIGN_IDENTITY_REGEXP="https://github.com/researchwiseai/pulse-py/.*"
COSIGN_OIDC_ISSUER="https://token.actions.githubusercontent.com"

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Pulse SDK Supply Chain Security Verification${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."

    # Check if cosign is installed
    if ! command -v cosign &> /dev/null; then
        print_error "cosign is not installed. Installing..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
            sudo mv cosign-linux-amd64 /usr/local/bin/cosign
            sudo chmod +x /usr/local/bin/cosign
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install cosign
            else
                curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-darwin-amd64"
                sudo mv cosign-darwin-amd64 /usr/local/bin/cosign
                sudo chmod +x /usr/local/bin/cosign
            fi
        else
            print_error "Unsupported OS. Please install cosign manually."
            exit 1
        fi
    fi

    # Check if syft is installed (optional, for SBOM verification)
    if ! command -v syft &> /dev/null; then
        print_info "syft is not installed. SBOM content verification will be skipped."
        print_info "To install syft: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
    fi

    print_success "Dependencies checked"
}

download_release_artifacts() {
    local version=$1
    print_step "Downloading release artifacts for version $version..."

    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    # Download release artifacts
    local release_url="https://github.com/$GITHUB_REPO/releases/download/$version"

    print_info "Downloading from: $release_url"

    # Download main artifacts
    curl -L -O "$release_url/pulse_sdk-${version#v}-py3-none-any.whl" || {
        print_error "Failed to download wheel file"
        return 1
    }

    curl -L -O "$release_url/pulse-sdk-${version#v}.tar.gz" || {
        print_error "Failed to download source distribution"
        return 1
    }

    # Download signatures and certificates
    curl -L -O "$release_url/pulse_sdk-${version#v}-py3-none-any.whl.sig" || {
        print_error "Failed to download wheel signature"
        return 1
    }

    curl -L -O "$release_url/pulse_sdk-${version#v}-py3-none-any.whl.crt" || {
        print_error "Failed to download wheel certificate"
        return 1
    }

    curl -L -O "$release_url/pulse-sdk-${version#v}.tar.gz.sig" || {
        print_error "Failed to download source signature"
        return 1
    }

    curl -L -O "$release_url/pulse-sdk-${version#v}.tar.gz.crt" || {
        print_error "Failed to download source certificate"
        return 1
    }

    # Download SBOMs
    curl -L -O "$release_url/sbom-wheel.spdx.json" || print_info "SBOM files may not be available for this release"
    curl -L -O "$release_url/sbom-source.spdx.json" || true
    curl -L -O "$release_url/sbom-wheel.cyclonedx.json" || true
    curl -L -O "$release_url/sbom-source.cyclonedx.json" || true

    # Download SBOM signatures if available
    curl -L -O "$release_url/sbom-wheel.spdx.json.sig" || true
    curl -L -O "$release_url/sbom-wheel.spdx.json.crt" || true

    # Download build provenance
    curl -L -O "$release_url/build-provenance.json" || print_info "Build provenance may not be available for this release"

    echo "$temp_dir"
}

verify_signatures() {
    local dir=$1
    print_step "Verifying digital signatures..."

    cd "$dir"

    # Verify wheel signature
    if [[ -f "pulse_sdk-"*"-py3-none-any.whl" && -f "pulse_sdk-"*"-py3-none-any.whl.sig" && -f "pulse_sdk-"*"-py3-none-any.whl.crt" ]]; then
        local wheel_file=$(ls pulse_sdk-*-py3-none-any.whl)
        print_info "Verifying signature for $wheel_file"

        if cosign verify-blob \
            --certificate "${wheel_file}.crt" \
            --signature "${wheel_file}.sig" \
            --certificate-identity-regexp "$COSIGN_IDENTITY_REGEXP" \
            --certificate-oidc-issuer "$COSIGN_OIDC_ISSUER" \
            "$wheel_file"; then
            print_success "Wheel signature verified"
        else
            print_error "Wheel signature verification failed"
            return 1
        fi
    else
        print_error "Missing wheel signature files"
        return 1
    fi

    # Verify source distribution signature
    if [[ -f "pulse-sdk-"*".tar.gz" && -f "pulse-sdk-"*".tar.gz.sig" && -f "pulse-sdk-"*".tar.gz.crt" ]]; then
        local source_file=$(ls pulse-sdk-*.tar.gz)
        print_info "Verifying signature for $source_file"

        if cosign verify-blob \
            --certificate "${source_file}.crt" \
            --signature "${source_file}.sig" \
            --certificate-identity-regexp "$COSIGN_IDENTITY_REGEXP" \
            --certificate-oidc-issuer "$COSIGN_OIDC_ISSUER" \
            "$source_file"; then
            print_success "Source distribution signature verified"
        else
            print_error "Source distribution signature verification failed"
            return 1
        fi
    else
        print_error "Missing source distribution signature files"
        return 1
    fi

    # Verify SBOM signatures if available
    if [[ -f "sbom-wheel.spdx.json" && -f "sbom-wheel.spdx.json.sig" && -f "sbom-wheel.spdx.json.crt" ]]; then
        print_info "Verifying SBOM signature"

        if cosign verify-blob \
            --certificate "sbom-wheel.spdx.json.crt" \
            --signature "sbom-wheel.spdx.json.sig" \
            --certificate-identity-regexp "$COSIGN_IDENTITY_REGEXP" \
            --certificate-oidc-issuer "$COSIGN_OIDC_ISSUER" \
            "sbom-wheel.spdx.json"; then
            print_success "SBOM signature verified"
        else
            print_error "SBOM signature verification failed"
            return 1
        fi
    else
        print_info "SBOM signature files not available"
    fi
}

verify_sbom_content() {
    local dir=$1
    print_step "Verifying SBOM content..."

    cd "$dir"

    if ! command -v syft &> /dev/null; then
        print_info "Skipping SBOM content verification (syft not installed)"
        return 0
    fi

    # Generate SBOM for downloaded wheel and compare
    if [[ -f "pulse_sdk-"*"-py3-none-any.whl" && -f "sbom-wheel.spdx.json" ]]; then
        local wheel_file=$(ls pulse_sdk-*-py3-none-any.whl)
        print_info "Generating SBOM for verification: $wheel_file"

        syft "$wheel_file" -o spdx-json=verification-sbom.spdx.json

        # Compare key components (simplified comparison)
        local original_packages=$(jq -r '.packages[].name' sbom-wheel.spdx.json | sort)
        local verification_packages=$(jq -r '.packages[].name' verification-sbom.spdx.json | sort)

        if [[ "$original_packages" == "$verification_packages" ]]; then
            print_success "SBOM content verification passed"
        else
            print_error "SBOM content verification failed - package lists differ"
            print_info "This may be due to different syft versions or configurations"
        fi
    else
        print_info "SBOM content verification skipped (files not available)"
    fi
}

display_summary() {
    local dir=$1
    print_step "Displaying verification summary..."

    cd "$dir"

    echo
    echo -e "${BLUE}=== VERIFICATION SUMMARY ===${NC}"
    echo

    # List all files
    echo -e "${YELLOW}Downloaded files:${NC}"
    ls -la
    echo

    # Show checksums
    echo -e "${YELLOW}File checksums:${NC}"
    sha256sum * 2>/dev/null || true
    echo

    # Show SBOM summary if available
    if [[ -f "sbom-wheel.spdx.json" ]]; then
        echo -e "${YELLOW}SBOM Summary (wheel):${NC}"
        if command -v jq &> /dev/null; then
            echo "Total packages: $(jq '.packages | length' sbom-wheel.spdx.json)"
            echo "SPDX version: $(jq -r '.spdxVersion' sbom-wheel.spdx.json)"
            echo "Creation date: $(jq -r '.creationInfo.created' sbom-wheel.spdx.json)"
        else
            echo "Install 'jq' to see detailed SBOM information"
        fi
        echo
    fi

    # Show build provenance if available
    if [[ -f "build-provenance.json" ]]; then
        echo -e "${YELLOW}Build Provenance:${NC}"
        if command -v jq &> /dev/null; then
            echo "Build type: $(jq -r '.buildType' build-provenance.json)"
            echo "Builder ID: $(jq -r '.builder.id' build-provenance.json)"
            echo "Source URI: $(jq -r '.materials[0].uri' build-provenance.json)"
        else
            echo "Install 'jq' to see detailed provenance information"
        fi
        echo
    fi
}

cleanup() {
    local dir=$1
    if [[ -n "$dir" && -d "$dir" ]]; then
        print_info "Cleaning up temporary directory: $dir"
        rm -rf "$dir"
    fi
}

main() {
    local version=$1

    if [[ -z "$version" ]]; then
        echo "Usage: $0 <version>"
        echo "Example: $0 v1.0.0"
        exit 1
    fi

    print_header

    # Trap to cleanup on exit
    local temp_dir=""
    trap 'cleanup "$temp_dir"' EXIT

    check_dependencies

    temp_dir=$(download_release_artifacts "$version")
    if [[ $? -ne 0 ]]; then
        print_error "Failed to download release artifacts"
        exit 1
    fi

    verify_signatures "$temp_dir"
    if [[ $? -ne 0 ]]; then
        print_error "Signature verification failed"
        exit 1
    fi

    verify_sbom_content "$temp_dir"

    display_summary "$temp_dir"

    print_success "Supply chain verification completed successfully!"
    print_info "Temporary files are in: $temp_dir"
    print_info "Files will be cleaned up on script exit"
}

# Run main function with all arguments
main "$@"
