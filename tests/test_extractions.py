import pytest
import httpx

from pulse.core.client import CoreClient


def test_extractions_not_implemented():
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    client = CoreClient(
        client=httpx.Client(transport=transport, base_url="https://api.example.com")
    )
    with pytest.raises(NotImplementedError):
        client.extract_elements(texts=["a"], categories=["b"])
