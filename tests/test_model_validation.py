"""Unit tests for updated Pydantic models in OpenAPI v0.9.0."""

import pytest
from pydantic import ValidationError
from pulse.core.models import (
    ThemesRequest,
    ThemeSetsResponse,
    Theme,
    SimilarityRequest,
    UnitAgg,
    Split,
    ClusteringRequest,
    ExtractionsRequest,
    UsageEstimateRequest,
    UsageEstimateResponse,
    UsageRecord,
    ErrorDetail,
    ErrorResponse,
)


class TestThemesRequest:
    """Test ThemesRequest model validation including cross-field constraints."""

    def test_valid_themes_request(self):
        """Test valid ThemesRequest creation."""
        request = ThemesRequest(
            inputs=["text1", "text2"],
            minThemes=2,
            maxThemes=10,
            context="test context",
            version="2025-09-01",
            prune=5,
            interactive=True,
            initialSets=2,
            fast=True,
        )
        assert request.inputs == ["text1", "text2"]
        assert request.minThemes == 2
        assert request.maxThemes == 10
        assert request.context == "test context"
        assert request.version == "2025-09-01"
        assert request.prune == 5
        assert request.interactive is True
        assert request.initialSets == 2
        assert request.fast is True

    def test_minimal_themes_request(self):
        """Test ThemesRequest with only required fields."""
        request = ThemesRequest(inputs=["text1", "text2"])
        assert request.inputs == ["text1", "text2"]
        assert request.minThemes is None
        assert request.maxThemes is None
        assert request.context is None
        assert request.version is None
        assert request.prune is None
        assert request.interactive is None
        assert request.initialSets is None
        assert request.fast is None

    def test_inputs_validation(self):
        """Test inputs field validation."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["single"])
        assert "at least 2" in str(exc_info.value)

        # Test maximum length
        long_inputs = [f"text{i}" for i in range(501)]
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=long_inputs)
        assert "at most 500" in str(exc_info.value)

    def test_min_themes_validation(self):
        """Test minThemes field validation."""
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], minThemes=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_themes_validation(self):
        """Test maxThemes field validation."""
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], maxThemes=51)
        assert "less than or equal to 50" in str(exc_info.value)

    def test_prune_validation(self):
        """Test prune field validation."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], prune=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], prune=26)
        assert "less than or equal to 25" in str(exc_info.value)

    def test_initial_sets_validation(self):
        """Test initialSets field validation."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], initialSets=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], initialSets=4)
        assert "less than or equal to 3" in str(exc_info.value)

    def test_initial_sets_interactive_constraint(self):
        """Test that initialSets > 1 requires interactive=true."""
        # Valid: initialSets=1 without interactive
        request = ThemesRequest(inputs=["text1", "text2"], initialSets=1)
        assert request.initialSets == 1
        assert request.interactive is None

        # Valid: initialSets > 1 with interactive=True
        request = ThemesRequest(
            inputs=["text1", "text2"], initialSets=2, interactive=True
        )
        assert request.initialSets == 2
        assert request.interactive is True

        # Invalid: initialSets > 1 without interactive=True
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], initialSets=2)
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)

        # Invalid: initialSets > 1 with interactive=False
        with pytest.raises(ValidationError) as exc_info:
            ThemesRequest(inputs=["text1", "text2"], initialSets=2, interactive=False)
        assert "initialSets > 1 requires interactive=true" in str(exc_info.value)


class TestTheme:
    """Test Theme model with new fields."""

    def test_valid_theme(self):
        """Test valid Theme creation."""
        theme = Theme(
            shortLabel="AI Tech",
            label="Artificial Intelligence Technology",
            description="Themes related to AI and machine learning technologies.",
            representatives=[
                "AI is transforming industries",
                "Machine learning advances",
            ],
        )
        assert theme.shortLabel == "AI Tech"
        assert theme.label == "Artificial Intelligence Technology"
        assert (
            theme.description
            == "Themes related to AI and machine learning technologies."
        )
        assert theme.representatives == [
            "AI is transforming industries",
            "Machine learning advances",
        ]

    def test_representatives_validation(self):
        """Test representatives field validation."""
        # Test minimum length (now allows 1-10)
        theme = Theme(
            shortLabel="Test",
            label="Test Theme",
            description="Test description",
            representatives=["single"],
        )
        assert len(theme.representatives) == 1

        # Test maximum length
        with pytest.raises(ValidationError) as exc_info:
            Theme(
                shortLabel="Test",
                label="Test Theme",
                description="Test description",
                representatives=["rep"] * 11,  # Too many
            )
        assert "at most 10" in str(exc_info.value)


class TestThemeSetsResponse:
    """Test ThemeSetsResponse model for version 2025-09-01."""

    def test_valid_theme_sets_response(self):
        """Test valid ThemeSetsResponse creation."""
        theme1 = Theme(
            shortLabel="Tech",
            label="Technology",
            description="Technology themes",
            representatives=["tech1", "tech2"],
        )
        theme2 = Theme(
            shortLabel="Health",
            label="Healthcare",
            description="Healthcare themes",
            representatives=["health1", "health2"],
        )

        response = ThemeSetsResponse(
            themeSets=[[theme1], [theme2]], requestId="test-request-id"
        )
        assert len(response.themeSets) == 2
        assert len(response.themeSets[0]) == 1
        assert len(response.themeSets[1]) == 1
        assert response.requestId == "test-request-id"

    def test_theme_sets_max_length(self):
        """Test themeSets maximum length validation."""
        theme = Theme(
            shortLabel="Test",
            label="Test Theme",
            description="Test description",
            representatives=["rep1", "rep2"],
        )

        # Valid: 3 theme sets
        response = ThemeSetsResponse(themeSets=[[theme], [theme], [theme]])
        assert len(response.themeSets) == 3

        # Invalid: 4 theme sets
        with pytest.raises(ValidationError) as exc_info:
            ThemeSetsResponse(themeSets=[[theme], [theme], [theme], [theme]])
        assert "at most 3" in str(exc_info.value)


class TestUnitAgg:
    """Test UnitAgg model for text splitting."""

    def test_valid_unit_agg(self):
        """Test valid UnitAgg creation."""
        unit_agg = UnitAgg(unit="sentence", agg="mean", window_size=3, stride_size=2)
        assert unit_agg.unit == "sentence"
        assert unit_agg.agg == "mean"
        assert unit_agg.window_size == 3
        assert unit_agg.stride_size == 2

    def test_default_values(self):
        """Test UnitAgg default values."""
        unit_agg = UnitAgg(unit="word")
        assert unit_agg.unit == "word"
        assert unit_agg.agg == "mean"
        assert unit_agg.window_size == 1
        assert unit_agg.stride_size == 1

    def test_unit_validation(self):
        """Test unit field validation."""
        # Valid units
        for unit in ["sentence", "newline", "word"]:
            unit_agg = UnitAgg(unit=unit)
            assert unit_agg.unit == unit

        # Invalid unit
        with pytest.raises(ValidationError) as exc_info:
            UnitAgg(unit="invalid")
        assert "Input should be 'sentence', 'newline' or 'word'" in str(exc_info.value)

    def test_agg_validation(self):
        """Test agg field validation."""
        # Valid aggregations
        for agg in ["mean", "max", "top2", "top3"]:
            unit_agg = UnitAgg(unit="sentence", agg=agg)
            assert unit_agg.agg == agg

        # Invalid aggregation
        with pytest.raises(ValidationError) as exc_info:
            UnitAgg(unit="sentence", agg="invalid")
        assert "Input should be 'mean', 'max', 'top2' or 'top3'" in str(exc_info.value)

    def test_window_size_validation(self):
        """Test window_size field validation."""
        with pytest.raises(ValidationError) as exc_info:
            UnitAgg(unit="sentence", window_size=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_stride_size_validation(self):
        """Test stride_size field validation."""
        with pytest.raises(ValidationError) as exc_info:
            UnitAgg(unit="sentence", stride_size=0)
        assert "greater than or equal to 1" in str(exc_info.value)


class TestSplit:
    """Test Split model for similarity requests."""

    def test_valid_split(self):
        """Test valid Split creation."""
        unit_agg_a = UnitAgg(unit="sentence", agg="mean")
        unit_agg_b = UnitAgg(unit="word", agg="max")

        split = Split(set_a=unit_agg_a, set_b=unit_agg_b)
        assert split.set_a == unit_agg_a
        assert split.set_b == unit_agg_b

    def test_optional_fields(self):
        """Test Split with optional fields."""
        split = Split()
        assert split.set_a is None
        assert split.set_b is None

        split = Split(set_a=UnitAgg(unit="sentence"))
        assert split.set_a is not None
        assert split.set_b is None


class TestSimilarityRequest:
    """Test SimilarityRequest with splitting configuration."""

    def test_valid_self_similarity_request(self):
        """Test valid self-similarity request."""
        request = SimilarityRequest(
            set=["text1", "text2", "text3"],
            fast=True,
            flatten=True,
            version="v1",
        )
        assert request.set == ["text1", "text2", "text3"]
        assert request.set_a is None
        assert request.set_b is None
        assert request.fast is True
        assert request.flatten is True
        assert request.version == "v1"

    def test_valid_cross_similarity_request(self):
        """Test valid cross-similarity request."""
        request = SimilarityRequest(
            set_a=["text1", "text2"],
            set_b=["text3", "text4"],
            fast=False,
            flatten=False,
        )
        assert request.set is None
        assert request.set_a == ["text1", "text2"]
        assert request.set_b == ["text3", "text4"]
        assert request.fast is False
        assert request.flatten is False

    def test_similarity_request_with_split(self):
        """Test SimilarityRequest with split configuration."""
        split = Split(
            set_a=UnitAgg(unit="sentence", agg="mean"),
            set_b=UnitAgg(unit="word", agg="max"),
        )
        request = SimilarityRequest(
            set_a=["text1", "text2"], set_b=["text3", "text4"], split=split
        )
        assert request.split == split

    def test_set_validation_constraints(self):
        """Test set validation constraints."""
        # Invalid: no sets provided
        with pytest.raises(ValidationError) as exc_info:
            SimilarityRequest()
        assert "Provide `set` or both `set_a` and `set_b`" in str(exc_info.value)

        # Invalid: only set_a provided
        with pytest.raises(ValidationError) as exc_info:
            SimilarityRequest(set_a=["text1"])
        assert "Provide `set` or both `set_a` and `set_b`" in str(exc_info.value)

        # Invalid: both set and set_a provided
        with pytest.raises(ValidationError) as exc_info:
            SimilarityRequest(set=["text1", "text2"], set_a=["text3"])
        assert "Cannot provide both `set` and `set_a`/`set_b`" in str(exc_info.value)

    def test_set_length_validation(self):
        """Test set length validation."""
        # Test minimum length for set
        with pytest.raises(ValidationError) as exc_info:
            SimilarityRequest(set=["single"])
        assert "at least 2" in str(exc_info.value)

        # Test maximum length for set
        long_set = [f"text{i}" for i in range(44722)]
        with pytest.raises(ValidationError) as exc_info:
            SimilarityRequest(set=long_set)
        assert "at most 44721" in str(exc_info.value)


class TestClusteringRequest:
    """Test ClusteringRequest with algorithm validation."""

    def test_valid_clustering_request(self):
        """Test valid ClusteringRequest creation."""
        request = ClusteringRequest(
            inputs=["text1", "text2", "text3"],
            k=3,
            algorithm="kmeans",
            fast=True,
        )
        assert request.inputs == ["text1", "text2", "text3"]
        assert request.k == 3
        assert request.algorithm == "kmeans"
        assert request.fast is True

    def test_default_algorithm(self):
        """Test default algorithm value."""
        request = ClusteringRequest(inputs=["text1", "text2"], k=2)
        assert request.algorithm == "kmeans"

    def test_algorithm_validation(self):
        """Test algorithm field validation."""
        # Valid algorithms
        for algorithm in ["kmeans", "skmeans", "agglomerative", "hdbscan"]:
            request = ClusteringRequest(
                inputs=["text1", "text2"], k=2, algorithm=algorithm
            )
            assert request.algorithm == algorithm

        # Invalid algorithm
        with pytest.raises(ValidationError) as exc_info:
            ClusteringRequest(inputs=["text1", "text2"], k=2, algorithm="invalid")
        assert (
            "Input should be 'kmeans', 'skmeans', 'agglomerative' or 'hdbscan'"
            in str(exc_info.value)
        )

    def test_inputs_validation(self):
        """Test inputs field validation."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            ClusteringRequest(inputs=["single"], k=1)
        assert "at least 2" in str(exc_info.value)

        # Test maximum length
        long_inputs = [f"text{i}" for i in range(44722)]
        with pytest.raises(ValidationError) as exc_info:
            ClusteringRequest(inputs=long_inputs, k=2)
        assert "at most 44721" in str(exc_info.value)

    def test_k_validation(self):
        """Test k field validation."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            ClusteringRequest(inputs=["text1", "text2"], k=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            ClusteringRequest(inputs=["text1", "text2"], k=51)
        assert "less than or equal to 50" in str(exc_info.value)


class TestExtractionsRequest:
    """Test ExtractionsRequest with type constraints."""

    def test_valid_extractions_request(self):
        """Test valid ExtractionsRequest creation."""
        request = ExtractionsRequest(
            inputs=["text1", "text2"],
            dictionary=["term1", "term2", "term3"],
            type="named-entities",
            expand_dictionary=True,
            expand_dictionary_limit=10,
            version="v1",
            fast=True,
        )
        assert request.inputs == ["text1", "text2"]
        assert request.dictionary == ["term1", "term2", "term3"]
        assert request.type == "named-entities"
        assert request.expand_dictionary is True
        assert request.expand_dictionary_limit == 10
        assert request.version == "v1"
        assert request.fast is True

    def test_default_type(self):
        """Test default type value."""
        request = ExtractionsRequest(
            inputs=["text1"], dictionary=["term1", "term2", "term3"]
        )
        assert request.type == "named-entities"

    def test_type_validation(self):
        """Test type field validation."""
        # Valid types
        for type_val in ["named-entities", "themes"]:
            request = ExtractionsRequest(
                inputs=["text1"], dictionary=["term1", "term2", "term3"], type=type_val
            )
            assert request.type == type_val

        # Invalid type
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(
                inputs=["text1"], dictionary=["term1", "term2", "term3"], type="invalid"
            )
        assert "Input should be 'named-entities' or 'themes'" in str(exc_info.value)

    def test_themes_expand_dictionary_constraint(self):
        """Test that expand_dictionary=false when type='themes'."""
        # Valid: themes with expand_dictionary=False
        request = ExtractionsRequest(
            inputs=["text1"],
            dictionary=["term1", "term2", "term3"],
            type="themes",
            expand_dictionary=False,
        )
        assert request.type == "themes"
        assert request.expand_dictionary is False

        # Valid: themes with default expand_dictionary (False)
        request = ExtractionsRequest(
            inputs=["text1"], dictionary=["term1", "term2", "term3"], type="themes"
        )
        assert request.type == "themes"
        assert request.expand_dictionary is False

        # Invalid: themes with expand_dictionary=True
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(
                inputs=["text1"],
                dictionary=["term1", "term2", "term3"],
                type="themes",
                expand_dictionary=True,
            )
        assert "expand_dictionary must be false when type is 'themes'" in str(
            exc_info.value
        )

    def test_inputs_validation(self):
        """Test inputs field validation."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(inputs=[], dictionary=["term1", "term2", "term3"])
        assert "at least 1" in str(exc_info.value)

        # Test maximum length
        long_inputs = [f"text{i}" for i in range(5001)]
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(
                inputs=long_inputs, dictionary=["term1", "term2", "term3"]
            )
        assert "at most 5000" in str(exc_info.value)

    def test_dictionary_validation(self):
        """Test dictionary field validation."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(inputs=["text1"], dictionary=["term1", "term2"])
        assert "at least 3" in str(exc_info.value)

        # Test maximum length
        long_dict = [f"term{i}" for i in range(201)]
        with pytest.raises(ValidationError) as exc_info:
            ExtractionsRequest(inputs=["text1"], dictionary=long_dict)
        assert "at most 200" in str(exc_info.value)


class TestUsageEstimateModels:
    """Test UsageEstimateRequest and UsageEstimateResponse models."""

    def test_valid_usage_estimate_request(self):
        """Test valid UsageEstimateRequest creation."""
        request = UsageEstimateRequest(feature="embeddings", inputs=["text1", "text2"])
        assert request.feature == "embeddings"
        assert request.inputs == ["text1", "text2"]

    def test_feature_validation(self):
        """Test feature field validation."""
        # Valid features
        valid_features = [
            "embeddings",
            "sentiment",
            "themes",
            "extractions",
            "summaries",
            "clustering",
            "similarity",
        ]
        for feature in valid_features:
            request = UsageEstimateRequest(feature=feature, inputs=["text1"])
            assert request.feature == feature

        # Invalid feature
        with pytest.raises(ValidationError) as exc_info:
            UsageEstimateRequest(feature="invalid", inputs=["text1"])
        assert "Input should be" in str(exc_info.value)

    def test_inputs_validation(self):
        """Test inputs field validation."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            UsageEstimateRequest(feature="embeddings", inputs=[])
        assert "at least 1" in str(exc_info.value)

    def test_valid_usage_estimate_response(self):
        """Test valid UsageEstimateResponse creation."""
        response = UsageEstimateResponse(
            usage={
                "total": 100,
                "records": [{"feature": "embeddings", "quantity": 100}],
            }
        )
        assert response.usage["total"] == 100
        assert len(response.usage["records"]) == 1


class TestUsageRecord:
    """Test UsageRecord model with backward compatibility."""

    def test_valid_usage_record(self):
        """Test valid UsageRecord creation."""
        record = UsageRecord(feature="embeddings", quantity=100)
        assert record.feature == "embeddings"
        assert record.quantity == 100

    def test_backward_compatibility_units_property(self):
        """Test backward compatibility units property."""
        record = UsageRecord(feature="embeddings", quantity=100)
        assert record.units == 100  # Should return quantity value


class TestErrorModels:
    """Test ErrorDetail and ErrorResponse models."""

    def test_valid_error_detail(self):
        """Test valid ErrorDetail creation."""
        error_detail = ErrorDetail(
            code="VALIDATION_ERROR",
            message="Field validation failed",
            path=["inputs", 0],
            field="inputs",
            location="body",
        )
        assert error_detail.code == "VALIDATION_ERROR"
        assert error_detail.message == "Field validation failed"
        assert error_detail.path == ["inputs", 0]
        assert error_detail.field == "inputs"
        assert error_detail.location == "body"

    def test_minimal_error_detail(self):
        """Test ErrorDetail with only required fields."""
        error_detail = ErrorDetail(message="Error occurred")
        assert error_detail.message == "Error occurred"
        assert error_detail.code is None
        assert error_detail.path is None
        assert error_detail.field is None
        assert error_detail.location is None

    def test_location_validation(self):
        """Test location field validation."""
        # Valid locations
        for location in ["body", "query", "header", "path"]:
            error_detail = ErrorDetail(message="Error", location=location)
            assert error_detail.location == location

        # Invalid location
        with pytest.raises(ValidationError) as exc_info:
            ErrorDetail(message="Error", location="invalid")
        assert "Input should be 'body', 'query', 'header' or 'path'" in str(
            exc_info.value
        )

    def test_valid_error_response(self):
        """Test valid ErrorResponse creation."""
        error_detail = ErrorDetail(message="Field error", field="inputs")
        error_response = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            errors=[error_detail],
            meta={"request_id": "req123"},
        )
        assert error_response.code == "VALIDATION_ERROR"
        assert error_response.message == "Request validation failed"
        assert len(error_response.errors) == 1
        assert error_response.meta["request_id"] == "req123"

    def test_minimal_error_response(self):
        """Test ErrorResponse with only required fields."""
        error_response = ErrorResponse(
            code="GENERIC_ERROR", message="Something went wrong"
        )
        assert error_response.code == "GENERIC_ERROR"
        assert error_response.message == "Something went wrong"
        assert error_response.errors is None
        assert error_response.meta is None
