"""Validation and error handling tests for OpenAPI v0.9.0 updates."""

import pytest
import httpx
from pulse.core.client import CoreClient
from pulse.core.exceptions import PulseAPIError
from pulse.core.validation import PulseValidationError, validate_before_request


class TestInputLimitValidation:
    """Test input limit validation for all endpoints."""

    def test_embeddings_sync_limit_validation(self):
        """Test embeddings sync mode input limit (200)."""
        CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(200)]
        try:
            validate_before_request("embeddings", inputs=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(201)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("embeddings", inputs=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "200" in str(exc_info.value)

    def test_embeddings_async_limit_validation(self):
        """Test embeddings async mode input limit (5000)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(5000)]
        try:
            validate_before_request("embeddings", inputs=inputs_at_limit, fast=False)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(5001)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("embeddings", inputs=inputs_over_limit, fast=False)
        assert "async mode" in str(exc_info.value)
        assert "5000" in str(exc_info.value)

    def test_similarity_self_sync_limit_validation(self):
        """Test similarity self-similarity sync mode limit (500)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(500)]
        try:
            validate_before_request("similarity", set=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(501)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("similarity", set=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    def test_similarity_cross_sync_limit_validation(self):
        """Test similarity cross-similarity sync mode limit (|set_a|×|set_b| ≤ 20,000)."""  # noqa: E501
        # Test at limit (should pass validation)
        set_a_at_limit = [f"text_a{i}" for i in range(100)]  # 100 * 200 = 20,000
        set_b_at_limit = [f"text_b{i}" for i in range(200)]
        try:
            validate_before_request(
                "similarity", set_a=set_a_at_limit, set_b=set_b_at_limit, fast=True
            )
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        set_a_over_limit = [f"text_a{i}" for i in range(101)]  # 101 * 200 = 20,200
        set_b_over_limit = [f"text_b{i}" for i in range(200)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request(
                "similarity", set_a=set_a_over_limit, set_b=set_b_over_limit, fast=True
            )
        assert "Cross-product too large" in str(exc_info.value)
        assert "20000" in str(exc_info.value) or "20,000" in str(exc_info.value)

    def test_themes_sync_limit_validation(self):
        """Test themes sync mode input limit (200)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(200)]
        try:
            validate_before_request("themes", inputs=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(201)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("themes", inputs=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "200" in str(exc_info.value)

    def test_themes_async_limit_validation(self):
        """Test themes async mode input limit (500)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(500)]
        try:
            validate_before_request("themes", inputs=inputs_at_limit, fast=False)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(501)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("themes", inputs=inputs_over_limit, fast=False)
        assert "async mode" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    def test_clustering_sync_limit_validation(self):
        """Test clustering sync mode input limit (500)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(500)]
        try:
            validate_before_request("clustering", inputs=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(501)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("clustering", inputs=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "500" in str(exc_info.value)

    def test_clustering_async_limit_validation(self):
        """Test clustering async mode input limit (44,721)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(44721)]
        try:
            validate_before_request("clustering", inputs=inputs_at_limit, fast=False)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(44722)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("clustering", inputs=inputs_over_limit, fast=False)
        assert "async mode" in str(exc_info.value)
        assert "44721" in str(exc_info.value)

    def test_sentiment_sync_limit_validation(self):
        """Test sentiment sync mode input limit (200)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(200)]
        try:
            validate_before_request("sentiment", inputs=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(201)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("sentiment", inputs=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "200" in str(exc_info.value)

    def test_sentiment_async_limit_validation(self):
        """Test sentiment async mode input limit (5000)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(5000)]
        try:
            validate_before_request("sentiment", inputs=inputs_at_limit, fast=False)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(5001)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("sentiment", inputs=inputs_over_limit, fast=False)
        assert "async mode" in str(exc_info.value)
        assert "5000" in str(exc_info.value)

    def test_extractions_sync_limit_validation(self):
        """Test extractions sync mode input limit (200)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(200)]
        try:
            validate_before_request("extractions", inputs=inputs_at_limit, fast=True)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(201)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("extractions", inputs=inputs_over_limit, fast=True)
        assert "sync mode" in str(exc_info.value)
        assert "200" in str(exc_info.value)

    def test_extractions_async_limit_validation(self):
        """Test extractions async mode input limit (5000)."""
        # Test at limit (should pass validation)
        inputs_at_limit = [f"text{i}" for i in range(5000)]
        try:
            validate_before_request("extractions", inputs=inputs_at_limit, fast=False)
        except PulseValidationError:
            pytest.fail("Validation should pass at limit")

        # Test over limit (should fail validation)
        inputs_over_limit = [f"text{i}" for i in range(5001)]
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("extractions", inputs=inputs_over_limit, fast=False)
        assert "async mode" in str(exc_info.value)
        assert "5000" in str(exc_info.value)


class TestCrossFieldValidationConstraints:
    """Test cross-field validation constraints."""

    def test_themes_initial_sets_interactive_constraint(self):
        """Test that initialSets > 1 requires interactive=true."""
        # Valid: initialSets=1 without interactive
        try:
            validate_before_request("themes", inputs=["text1", "text2"], initialSets=1)
        except PulseValidationError:
            pytest.fail("Validation should pass for initialSets=1 without interactive")

        # Valid: initialSets > 1 with interactive=True
        try:
            validate_before_request(
                "themes", inputs=["text1", "text2"], initialSets=2, interactive=True
            )
        except PulseValidationError:
            pytest.fail(
                "Validation should pass for initialSets > 1 with interactive=True"
            )

        # Invalid: initialSets > 1 without interactive=True
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("themes", inputs=["text1", "text2"], initialSets=2)
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)

        # Invalid: initialSets > 1 with interactive=False
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request(
                "themes", inputs=["text1", "text2"], initialSets=2, interactive=False
            )
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)

    def test_similarity_set_constraints(self):
        """Test similarity set validation constraints."""
        # Invalid: no sets provided
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("similarity")
        expected_msg = (
            "Provide 'set' for self-similarity or both 'set_a' and 'set_b' "
            "for cross-similarity"
        )
        assert expected_msg in str(exc_info.value)

        # Invalid: only set_a provided
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request("similarity", set_a=["text1"])
        expected_msg = (
            "Provide 'set' for self-similarity or both 'set_a' and 'set_b' "
            "for cross-similarity"
        )
        assert expected_msg in str(exc_info.value)

        # Invalid: both set and set_a provided
        with pytest.raises(PulseValidationError) as exc_info:
            validate_before_request(
                "similarity", set=["text1", "text2"], set_a=["text3"]
            )
        assert "Cannot provide both 'set' and 'set_a'/'set_b'" in str(exc_info.value)

        # Valid: set provided
        try:
            validate_before_request("similarity", set=["text1", "text2"])
        except PulseValidationError:
            pytest.fail("Validation should pass with set provided")

        # Valid: both set_a and set_b provided
        try:
            validate_before_request("similarity", set_a=["text1"], set_b=["text2"])
        except PulseValidationError:
            pytest.fail("Validation should pass with both set_a and set_b provided")


class TestErrorResponseParsing:
    """Test error response parsing and handling."""

    def test_validation_error_parsing(self):
        """Test parsing of validation error responses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "errors": [
                        {
                            "code": "too_small",
                            "message": "List should have at least 2 items after validation, not 1",  # noqa: E501
                            "path": ["inputs"],
                            "field": "inputs",
                            "location": "body",
                        },
                        {
                            "code": "invalid_choice",
                            "message": "Input should be 'kmeans', 'skmeans', 'agglomerative' or 'hdbscan'",  # noqa: E501
                            "path": ["algorithm"],
                            "field": "algorithm",
                            "location": "body",
                        },
                    ],
                    "meta": {
                        "request_id": "req_123",
                        "timestamp": "2025-01-01T00:00:00Z",
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        # The client-side validation will catch the invalid algorithm before API call
        with pytest.raises(ValueError) as exc_info:
            core_client.cluster_texts(inputs=["single"], k=1, algorithm="invalid")

        assert "algorithm must be one of" in str(exc_info.value)

    def test_credit_error_parsing(self):
        """Test parsing of credit error responses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={
                    "code": "insufficient_credits",
                    "message": "Insufficient credits to complete request",
                    "meta": {
                        "required_credits": 100,
                        "available_credits": 50,
                        "account_id": "acc_123",
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        with pytest.raises(PulseAPIError) as exc_info:
            core_client.generate_themes(["text1", "text2"])

        error = exc_info.value
        assert error.status == 402
        assert "insufficient_credits" in str(error)
        assert "Insufficient credits" in str(error)

    def test_authentication_error_parsing(self):
        """Test parsing of authentication error responses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "code": "unauthorized",
                    "message": "Invalid or expired authentication token",
                    "meta": {
                        "auth_type": "bearer",
                        "expires_at": "2025-01-01T00:00:00Z",
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        with pytest.raises(PulseAPIError) as exc_info:
            core_client.generate_themes(["text1", "text2"])

        error = exc_info.value
        assert error.status == 401
        assert "unauthorized" in str(error)
        assert "Invalid or expired authentication token" in str(error)

    def test_rate_limit_error_parsing(self):
        """Test parsing of rate limit error responses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Please try again later.",
                    "meta": {"retry_after": 60, "limit": 100, "window": "1h"},
                },
                headers={"Retry-After": "60"},
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        with pytest.raises(PulseAPIError) as exc_info:
            core_client.generate_themes(["text1", "text2"])

        error = exc_info.value
        assert error.status == 429
        assert "rate_limit_exceeded" in str(error)
        assert "Rate limit exceeded" in str(error)

    def test_server_error_parsing(self):
        """Test parsing of server error responses."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                json={
                    "code": "internal_server_error",
                    "message": "An internal server error occurred",
                    "meta": {
                        "error_id": "err_123",
                        "timestamp": "2025-01-01T00:00:00Z",
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        with pytest.raises(PulseAPIError) as exc_info:
            core_client.generate_themes(["text1", "text2"])

        error = exc_info.value
        assert error.status == 500
        assert "internal_server_error" in str(error)
        assert "An internal server error occurred" in str(error)


class TestClientSideValidationHelpers:
    """Test client-side validation helpers."""

    def test_client_side_themes_validation(self):
        """Test client-side validation for themes endpoint."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test initialSets constraint validation
        with pytest.raises(ValueError) as exc_info:
            client.generate_themes(
                texts=["text1", "text2"], initial_sets=2, interactive=False
            )
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)

    def test_client_side_clustering_validation(self):
        """Test client-side validation for clustering endpoint."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test algorithm validation
        with pytest.raises(ValueError) as exc_info:
            client.cluster_texts(inputs=["text1", "text2"], k=1, algorithm="invalid")
        assert "algorithm must be one of" in str(exc_info.value)

    def test_client_side_similarity_validation(self):
        """Test client-side validation for similarity endpoint."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test set constraint validation
        with pytest.raises(ValueError) as exc_info:
            client.compare_similarity()
        expected_msg = (
            "Provide 'set' for self-similarity or both 'set_a' and 'set_b' "
            "for cross-similarity"
        )
        assert expected_msg in str(exc_info.value)

        # Test conflicting sets validation
        with pytest.raises(ValueError) as exc_info:
            client.compare_similarity(set=["text1", "text2"], set_a=["text3"])
        assert "Cannot provide both 'set' and 'set_a'/'set_b'" in str(exc_info.value)

    def test_client_side_extractions_validation(self):
        """Test client-side validation for extractions endpoint."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test type validation
        with pytest.raises(ValueError) as exc_info:
            client.extract_elements(
                inputs=["text1"], dictionary=["term1", "term2", "term3"], type="invalid"
            )
        assert "type must be one of" in str(exc_info.value)

    def test_client_side_usage_estimate_validation(self):
        """Test client-side validation for usage estimate endpoint."""
        client = CoreClient(client=httpx.Client(base_url="https://api.example.com"))

        # Test feature validation
        with pytest.raises(ValueError) as exc_info:
            client.estimate_usage(feature="invalid", inputs=["text1"])
        assert "feature must be one of" in str(exc_info.value)


class TestContextPreservation:
    """Test error context preservation for debugging."""

    def test_validation_error_context_preservation(self):
        """Test that validation errors preserve context for debugging."""
        try:
            validate_before_request(
                "themes", inputs=["text1", "text2"], initialSets=2, interactive=False
            )
        except PulseValidationError as e:
            # Check that error contains useful context
            assert e.endpoint == "themes"
            assert "initialSets > 1 requires interactive=true" in str(e)
            # The error should be informative enough for debugging
            assert "initialSets" in str(e)
            assert "interactive" in str(e)

    def test_api_error_context_preservation(self):
        """Test that API errors preserve context for debugging."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "code": "validation_error",
                    "message": "Field validation failed",
                    "errors": [
                        {
                            "code": "too_small",
                            "message": "List should have at least 2 items",
                            "path": ["inputs"],
                            "field": "inputs",
                            "location": "body",
                        }
                    ],
                },
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="https://api.example.com")
        core_client = CoreClient(client=client)

        try:
            core_client.cluster_texts(inputs=["single"], k=1)
        except PulseAPIError as e:
            # Check that error preserves useful context
            assert e.status == 400
            assert "validation_error" in str(e)
            assert "Field validation failed" in str(e)
            # Should contain the original error information
            assert hasattr(e, "code")
            assert e.code == "validation_error"
