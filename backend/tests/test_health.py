"""Tests for the health endpoint."""

from app.schemas import HealthResponse


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = HealthResponse.model_validate(response.json())
    assert data.status == "ok"
    assert data.service == "HerHealth AI"
    assert data.model_available is False


def test_any_origin_can_preflight_api_requests(client):
    response = client.options(
        "/api/dashboard",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "GET" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers