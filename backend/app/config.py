"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "HerHealth AI"
    app_env: str = "development"
    app_version: str = "0.1.0"

    # Local SQLite file for the prototype store. PostgreSQL can replace this later
    # because the data layer in app/database.py is router/repository separated.
    # Relative paths are resolved against the backend "data" directory.
    database_url: str = "sqlite:///herhealth.db"

    cors_origins: List[str] = [
        "https://herhealth-ai-frontend.onrender.com",
        "https://herhealth-ai-w4yw.onrender.com",
    ]

    # Screening image constraints.
    max_upload_size_mb: int = 5
    allowed_image_mime_types: List[str] = ["image/jpeg", "image/png", "image/webp"]

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()