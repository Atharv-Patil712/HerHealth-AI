import os
import tempfile

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///" + tempfile.mkdtemp() + "/test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from app.database import get_conn, init_db

    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS women_health_qa USING fts5(
                question,
                answer,
                tokenize='porter unicode61'
            )
            """
        )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client