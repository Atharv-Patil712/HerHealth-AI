"""Tests for the health endpoint."""

from fastapi.routing import APIRoute

from app.main import api
from app.schemas import HealthResponse

FRONTEND_ORIGIN = "https://herhealth-ai-frontend.onrender.com"


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = HealthResponse.model_validate(response.json())
    assert data.status == "ok"
    assert data.service == "HerHealth AI"
    assert data.model_available is False


def test_dashboard_endpoint_initializes_database(client):
    response = client.get("/api/dashboard")

    assert response.status_code == 200, response.text
    assert response.json()["welcome"]


def test_any_origin_can_preflight_api_requests(client):
    response = client.options(
        "/api/dashboard",
        headers={
            "Origin": FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
    assert "GET" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_unhandled_error_is_json_and_has_cors_headers(client):
    async def fail():
        raise RuntimeError("private diagnostic detail")

    route = APIRoute("/_test/unhandled-error", endpoint=fail, methods=["GET"])
    api.router.routes.append(route)
    try:
        response = client.get(
            "/_test/unhandled-error",
            headers={"Origin": FRONTEND_ORIGIN},
        )
    finally:
        api.router.routes.remove(route)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An unexpected server error occurred. Check backend logs for details."
    }
    assert response.headers["access-control-allow-origin"] == FRONTEND_ORIGIN