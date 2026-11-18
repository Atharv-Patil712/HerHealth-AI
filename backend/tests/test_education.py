"""Tests for the education Q&A search endpoint."""

from app.database import execute


def test_education_qa_search(client):
    execute(
        "INSERT INTO women_health_qa (question, answer) VALUES (?, ?)",
        ("What is anemia?", "Anemia is a condition involving low healthy red blood cells."),
    )

    response = client.get("/api/education/ask", params={"q": "what is anemia"})

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert any(result["question"] == "What is anemia?" for result in results)
