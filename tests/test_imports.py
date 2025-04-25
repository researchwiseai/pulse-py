"""Basic import tests."""


def test_version():
    import pulse_client

    assert pulse_client.__version__ == "0.1.0"
