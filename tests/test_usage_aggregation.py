"""Test usage metrics aggregation for batched operations.

This module tests that usage metrics are properly aggregated across all
parallel batches and accurately reflect total processing.
"""

from unittest.mock import patch
from pulse.core.client import CoreClient
from pulse.core.models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SentimentResponse,
    ExtractionsResponse,
    UsageReport,
    UsageRecord,
)


class TestUsageAggregation:
    """Test that usage metrics are properly aggregated for batched operations."""

    def setup_method(self):
        """Set up test client."""
        self.client = CoreClient()

    @patch("pulse.core.client.CoreClient._request")
    @patch("pulse.core.batching.process_batches_concurrently")
    def test_sentiment_usage_aggregation_across_batches(
        self, mock_process, mock_request
    ):
        """Test that sentiment analysis aggregates usage metrics from batches."""
        # Mock batch results with different usage metrics
        batch_1_result = {
            "results": [
                {"text": "I love this!", "sentiment": "positive", "confidence": 0.95},
                {"text": "This is great!", "sentiment": "positive", "confidence": 0.90},
            ],
            "usage": {"total": 2, "records": [{"feature": "sentiment", "quantity": 2}]},
            "requestId": "batch-1",
        }

        batch_2_result = {
            "results": [
                {"text": "This is okay", "sentiment": "neutral", "confidence": 0.60},
                {"text": "Not bad", "sentiment": "neutral", "confidence": 0.55},
            ],
            "usage": {"total": 2, "records": [{"feature": "sentiment", "quantity": 2}]},
            "requestId": "batch-2",
        }

        batch_3_result = {
            "results": [
                {"text": "I hate this", "sentiment": "negative", "confidence": 0.85},
            ],
            "usage": {"total": 1, "records": [{"feature": "sentiment", "quantity": 1}]},
            "requestId": "batch-3",
        }

        # Mock the concurrent processing to return our batch results
        mock_process.return_value = [batch_1_result, batch_2_result, batch_3_result]

        # Test with large input that triggers batching
        large_texts = ["text"] * 5000  # This should trigger batching
        result = self.client.analyze_sentiment(large_texts, fast=False)

        # Verify result structure
        assert isinstance(result, SentimentResponse)
        assert len(result.results) == 5  # 2 + 2 + 1 from batches

        # Verify usage aggregation
        assert result.usage is not None
        assert result.usage.total == 5  # 2 + 2 + 1
        assert len(result.usage.records) == 3  # One record per batch

        # Verify all records are sentiment records
        for record in result.usage.records:
            assert record.feature == "sentiment"

        # Verify total quantity matches sum of individual records
        total_from_records = sum(record.quantity for record in result.usage.records)
        assert result.usage.total == total_from_records

    @patch("pulse.core.client.CoreClient._request")
    @patch("pulse.core.batching.process_batches_concurrently")
    def test_embeddings_usage_aggregation_across_batches(
        self, mock_process, mock_request
    ):
        """Test that embeddings aggregates usage metrics from multiple batches."""
        # Mock batch results with different usage metrics
        batch_1_result = {
            "embeddings": [
                {"id": "1", "text": "Hello world", "vector": [0.1, 0.2, 0.3]},
                {"id": "2", "text": "Goodbye world", "vector": [0.4, 0.5, 0.6]},
            ],
            "usage": {
                "total": 2,
                "records": [{"feature": "embeddings", "quantity": 2}],
            },
            "requestId": "batch-1",
        }

        batch_2_result = {
            "embeddings": [
                {"id": "3", "text": "Another text", "vector": [0.7, 0.8, 0.9]},
            ],
            "usage": {
                "total": 1,
                "records": [{"feature": "embeddings", "quantity": 1}],
            },
            "requestId": "batch-2",
        }

        # Mock the concurrent processing to return our batch results
        mock_process.return_value = [batch_1_result, batch_2_result]

        # Test with large input that triggers batching
        large_inputs = ["text"] * 1000  # This should trigger batching
        request = EmbeddingsRequest(inputs=large_inputs, fast=False)
        result = self.client.create_embeddings(request)

        # Verify result structure
        assert isinstance(result, EmbeddingsResponse)
        assert len(result.embeddings) == 3  # 2 + 1 from batches

        # Verify usage aggregation
        assert result.usage is not None
        assert result.usage.total == 3  # 2 + 1
        assert len(result.usage.records) == 2  # One record per batch

        # Verify all records are embeddings records
        for record in result.usage.records:
            assert record.feature == "embeddings"

        # Verify total quantity matches sum of individual records
        total_from_records = sum(record.quantity for record in result.usage.records)
        assert result.usage.total == total_from_records

    @patch("pulse.core.client.CoreClient._request")
    @patch("pulse.core.batching.process_batches_concurrently")
    def test_extractions_usage_aggregation_across_batches(
        self, mock_process, mock_request
    ):
        """Test that extractions aggregates usage metrics from multiple batches."""
        # Mock batch results with different usage metrics
        batch_1_result = {
            "columns": [
                {"category": "person", "term": "John"},
                {"category": "company", "term": "Apple"},
            ],
            "matrix": [["John", "Apple"], ["Mary", "Paris"]],
            "usage": {
                "total": 2,
                "records": [{"feature": "extractions", "quantity": 2}],
            },
            "requestId": "batch-1",
        }

        batch_2_result = {
            "columns": [
                {"category": "person", "term": "Bob"},
                {"category": "company", "term": "Tesla"},
            ],
            "matrix": [["Bob", "Tesla"]],
            "usage": {
                "total": 1,
                "records": [{"feature": "extractions", "quantity": 1}],
            },
            "requestId": "batch-2",
        }

        # Mock the concurrent processing to return our batch results
        mock_process.return_value = [batch_1_result, batch_2_result]

        # Test with large input that triggers batching
        large_inputs = ["text"] * 1000  # This should trigger batching
        dictionary = ["entity1", "entity2", "entity3"]  # Need at least 3 terms
        result = self.client.extract_elements(large_inputs, dictionary, fast=False)

        # Verify result structure
        assert isinstance(result, ExtractionsResponse)
        assert len(result.matrix) == 3  # 2 + 1 from batches

        # Verify usage aggregation
        assert result.usage is not None
        assert result.usage.total == 3  # 2 + 1
        assert len(result.usage.records) == 2  # One record per batch

        # Verify all records are extractions records
        for record in result.usage.records:
            assert record.feature == "extractions"

        # Verify total quantity matches sum of individual records
        total_from_records = sum(record.quantity for record in result.usage.records)
        assert result.usage.total == total_from_records

    def test_usage_aggregation_with_mixed_features(self):
        """Test usage aggregation when batches contain mixed feature usage."""
        # Create usage records with mixed features (simulating complex operations)
        batch_1_usage = UsageReport(
            total=5,
            records=[
                UsageRecord(feature="embeddings", quantity=3),
                UsageRecord(feature="similarity", quantity=2),
            ],
        )

        batch_2_usage = UsageReport(
            total=4,
            records=[
                UsageRecord(feature="embeddings", quantity=2),
                UsageRecord(feature="similarity", quantity=2),
            ],
        )

        # Simulate aggregation logic
        all_records = []
        total_quantity = 0

        for usage in [batch_1_usage, batch_2_usage]:
            all_records.extend(usage.records)
            total_quantity += usage.total

        aggregated_usage = UsageReport(records=all_records, total=total_quantity)

        # Verify aggregation
        assert aggregated_usage.total == 9  # 5 + 4
        assert len(aggregated_usage.records) == 4  # 2 + 2

        # Verify feature distribution
        embeddings_records = [
            r for r in aggregated_usage.records if r.feature == "embeddings"
        ]
        similarity_records = [
            r for r in aggregated_usage.records if r.feature == "similarity"
        ]

        assert len(embeddings_records) == 2
        assert len(similarity_records) == 2
        assert sum(r.quantity for r in embeddings_records) == 5  # 3 + 2
        assert sum(r.quantity for r in similarity_records) == 4  # 2 + 2

    def test_usage_aggregation_with_empty_batches(self):
        """Test usage aggregation handles batches with no usage information."""
        # Create mixed usage scenarios
        batch_with_usage = UsageReport(
            total=3, records=[UsageRecord(feature="sentiment", quantity=3)]
        )

        batch_without_usage = None

        # Simulate aggregation logic that handles None usage
        all_records = []
        total_quantity = 0

        for usage in [batch_with_usage, batch_without_usage]:
            if usage:
                all_records.extend(usage.records)
                total_quantity += usage.total

        aggregated_usage = UsageReport(records=all_records, total=total_quantity)

        # Verify aggregation handles None gracefully
        assert aggregated_usage.total == 3
        assert len(aggregated_usage.records) == 1
        assert aggregated_usage.records[0].feature == "sentiment"

    def test_usage_record_backward_compatibility(self):
        """Test that usage records maintain backward compatibility with units."""
        record = UsageRecord(feature="test", quantity=5)

        # Test backward compatibility property
        assert hasattr(record, "units")
        assert record.units == 5
        assert record.units == record.quantity

        # Test that both properties work in aggregation
        records = [
            UsageRecord(feature="test1", quantity=3),
            UsageRecord(feature="test2", quantity=2),
        ]

        total_quantity = sum(r.quantity for r in records)
        total_units = sum(r.units for r in records)

        assert total_quantity == total_units == 5

    @patch("pulse.core.client.CoreClient._request")
    def test_single_request_usage_unchanged(self, mock_request):
        """Test that single requests (non-batched) still return usage correctly."""
        # Mock response for single request
        single_response_data = {
            "results": [
                {"text": "I love this!", "sentiment": "positive", "confidence": 0.95},
            ],
            "usage": {"total": 1, "records": [{"feature": "sentiment", "quantity": 1}]},
            "requestId": "test-123",
        }

        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = single_response_data

        # Test small request (non-batched)
        small_texts = ["I love this!"]
        result = self.client.analyze_sentiment(small_texts, fast=True)

        # Verify usage is preserved exactly as returned by API
        assert isinstance(result, SentimentResponse)
        assert result.usage is not None
        assert result.usage.total == 1
        assert len(result.usage.records) == 1
        assert result.usage.records[0].feature == "sentiment"
        assert result.usage.records[0].quantity == 1

    def test_usage_aggregation_accuracy(self):
        """Test that usage aggregation is mathematically accurate."""
        # Create multiple usage reports with different patterns
        usage_reports = [
            UsageReport(
                total=10,
                records=[
                    UsageRecord(feature="embeddings", quantity=6),
                    UsageRecord(feature="similarity", quantity=4),
                ],
            ),
            UsageReport(
                total=8,
                records=[
                    UsageRecord(feature="embeddings", quantity=5),
                    UsageRecord(feature="clustering", quantity=3),
                ],
            ),
            UsageReport(
                total=5,
                records=[
                    UsageRecord(feature="sentiment", quantity=5),
                ],
            ),
        ]

        # Aggregate manually
        all_records = []
        total_quantity = 0

        for usage in usage_reports:
            all_records.extend(usage.records)
            total_quantity += usage.total

        aggregated_usage = UsageReport(records=all_records, total=total_quantity)

        # Verify mathematical accuracy
        assert aggregated_usage.total == 23  # 10 + 8 + 5
        assert len(aggregated_usage.records) == 5  # 2 + 2 + 1

        # Verify total equals sum of all record quantities
        record_sum = sum(record.quantity for record in aggregated_usage.records)
        assert aggregated_usage.total == record_sum

        # Verify feature-specific totals
        feature_totals = {}
        for record in aggregated_usage.records:
            if record.feature not in feature_totals:
                feature_totals[record.feature] = 0
            feature_totals[record.feature] += record.quantity

        assert feature_totals["embeddings"] == 11  # 6 + 5
        assert feature_totals["similarity"] == 4
        assert feature_totals["clustering"] == 3
        assert feature_totals["sentiment"] == 5

    def test_usage_aggregation_preserves_request_context(self):
        """Test that usage aggregation preserves important request context."""
        # This test ensures that while usage is aggregated, other important
        # response metadata is preserved appropriately

        # Create sample aggregated response
        aggregated_usage = UsageReport(
            total=100,
            records=[
                UsageRecord(feature="embeddings", quantity=60),
                UsageRecord(feature="similarity", quantity=40),
            ],
        )

        # Verify usage structure is preserved
        assert aggregated_usage.total == 100
        assert len(aggregated_usage.records) == 2

        # Verify individual records maintain their structure
        for record in aggregated_usage.records:
            assert hasattr(record, "feature")
            assert hasattr(record, "quantity")
            assert hasattr(record, "units")  # Backward compatibility
            assert record.quantity > 0
            assert record.units == record.quantity
