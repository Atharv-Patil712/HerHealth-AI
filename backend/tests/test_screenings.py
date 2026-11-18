"""Tests for anemia screening upload validation."""

import io

from PIL import Image

from app.routers import screenings


def test_analyze_returns_screening_result(client, monkeypatch):
    monkeypatch.setattr(
        screenings,
        "unified_model",
        lambda _image, allow_center_fallback: {
            "roi_source": "conjunctiva",
            "seg_confidence": 0.95,
            "anemia_prob": 0.1,
            "hb_g_dl": 13.2,
            "roi_box": [100, 100, 540, 540],
        },
    )
    image = io.BytesIO()
    Image.new("RGB", (640, 640)).save(image, format="JPEG")
    image.seek(0)

    response = client.post(
        "/api/screenings/analyze",
        files={"file": ("eyelid.jpg", image, "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["roi_detected"] is True
    assert result["estimated_hemoglobin_g_dl"] == 13.2
    assert result["risk_tier"] == "Low"
    assert result["record_id"] is not None


def test_analyze_invalid_image_rejected(client):
    files = {"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/api/screenings/analyze", files=files)
    assert response.status_code == 400, response.text


def test_analyze_requires_file(client):
    response = client.post("/api/screenings/analyze")
    assert response.status_code == 422


def test_screenings_history(client):
    response = client.get("/api/screenings/history")
    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)