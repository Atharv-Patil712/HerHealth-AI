"""Tests for ensuring the Q&A knowledge base is built when absent."""

import sqlite3

from build_health_kb import ensure_knowledge_base


def test_ensure_knowledge_base_builds_only_when_missing(tmp_path, monkeypatch):
    database_path = tmp_path / "health.db"
    builds = []

    def build():
        builds.append(True)
        with sqlite3.connect(database_path) as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE women_health_qa USING fts5(
                    question,
                    answer,
                    tokenize='porter unicode61'
                )
                """
            )

    monkeypatch.setattr("build_health_kb.DB_PATH", database_path)
    monkeypatch.setattr("build_health_kb.build_knowledge_base", build)

    ensure_knowledge_base()
    ensure_knowledge_base()

    assert len(builds) == 1
