"""Configuration for the soccer analytics research package."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="SOCCER_EDGE_", env_file=".env", extra="ignore")

    env: str = "dev"
    data_dir: Path = Path("data")
    youtube_api_key: str | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    external_execution_enabled: bool = Field(default=False, validation_alias="SOCCER_EDGE_ENABLE_EXTERNAL_EXECUTION")


def get_settings() -> Settings:
    """Return configured settings."""

    return Settings()
