import os
import tempfile

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///" + tempfile.mkdtemp() + "/test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client