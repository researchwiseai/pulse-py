"""Basic import tests."""


def test_version():
    import pulse_client

    assert isinstance(pulse_client.__version__, str)
    assert len(pulse_client.__version__) > 0
