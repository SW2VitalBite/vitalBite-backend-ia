"""Fixtures compartidas de pytest."""

import os

import pytest
from fastapi.testclient import TestClient

# Asegura una API Key conocida para los tests antes de importar la app
os.environ.setdefault("API_KEY", "test_api_key")

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

API_KEY = settings.API_KEY


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY}
