"""Tests for anemia screening upload validation."""

import io

from PIL import Image, ImageDraw

from app.routers import screenings


def create_test_image():
    image = Image.new("RGB", (640, 640), (20, 120, 40))
    ImageDraw.Draw(image).rectangle((0, 0, 319, 639), fill=(180, 90, 30))
    return image


def test_analyze_returns_screening_result(client, monkeypatch):
    def fake_model(_image, allow_center_fallback):
        assert allow_center_fallback is False
        return {
            "roi_source": "conjunctiva",
            "seg_confidence": 0.95,
            "anemia_prob": 0.1,
            "hb_g_dl": 13.2,
            "roi_box": [100, 100, 540, 540],
        }

    monkeypatch.setattr(
        screenings,
        "unified_model",
        fake_model,
    )
    image = io.BytesIO()
    create_test_image().save(image, format="JPEG")
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


def test_analyze_rejects_image_without_conjunctiva(client, monkeypatch):
    monkeypatch.setattr(
        screenings,
        "unified_model",
        lambda _image, allow_center_fallback: {
            "roi_source": "none",
            "seg_confidence": 0.02,
            "mask_fraction": 0,
            "roi_box": None,
            "anemia_prob": None,
            "hb_g_dl": None,
        },
    )
    image = io.BytesIO()
    create_test_image().save(image, format="JPEG")
    image.seek(0)

    response = client.post(
        "/api/screenings/analyze",
        files={"file": ("other.jpg", image, "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "conjunctiva_not_found"
    assert "inner lower-eyelid photo" in response.json()["detail"]["message"]


def test_analyze_rejects_blank_image(client, monkeypatch):
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("The model must not analyze a blank image.")

    monkeypatch.setattr(screenings, "unified_model", should_not_run)
    image = io.BytesIO()
    Image.new("RGB", (640, 640), (0, 0, 0)).save(image, format="JPEG")
    image.seek(0)

    response = client.post(
        "/api/screenings/analyze",
        files={"file": ("black.jpg", image, "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "image_blank_or_too_dark"


def test_analyze_nail_rejects_blank_image(client, monkeypatch):
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("The nail model must not analyze a blank image.")

    monkeypatch.setattr(screenings, "predict_nail_anemia", should_not_run)
    image = io.BytesIO()
    Image.new("RGB", (640, 640), (0, 0, 0)).save(image, format="JPEG")
    image.seek(0)

    response = client.post(
        "/api/screenings/analyze-nail",
        files={"file": ("black.jpg", image, "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "image_blank_or_too_dark"


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