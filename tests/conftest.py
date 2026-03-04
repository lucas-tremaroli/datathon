import pytest
from fastapi.testclient import TestClient

from datathon.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_student():
    return {
        "ieg": 7.5,
        "iaa": 8.0,
        "ips": 6.5,
        "ida": 7.2,
        "ian": 8.5,
        "ipv": 6.0,
        "inde": 7.3,
        "stone": 3,
        "age": 14,
    }
