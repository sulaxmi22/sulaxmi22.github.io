"""
MetricsAI — Configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o-mini")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8002)
    cors_origins: list[str] = Field(default=["*"])
    stream_interval_ms: int = Field(default=1000, description="WebSocket push interval in ms")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
