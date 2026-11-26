import sys
sys.path.insert(0, '.')

import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
import screenings


def fake_unified_model(_image, allow_center_fallback):
    raise FileNotFoundError("Model file not found")


def test():
    screenings.unified_model = fake_unified_model
    
    image = io.BytesIO()
    img = Image.new("RGB", (640, 640), (20, 120, 40))
    img.save(image, format="JPEG")
    image.seek(0)
    
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/screenings/analyze",
        files={"file": ("eyelid.jpg", image, "image/jpeg")},
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response text: {response.text}")


if __name__ == "__main__":
    test()