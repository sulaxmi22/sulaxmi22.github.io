"""
AgentForge — Configuration
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o-mini")
    tavily_api_key: str = Field(default="")
    quality_threshold: float = Field(default=0.8)
    max_revisions: int = Field(default=3)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    cors_origins: list[str] = Field(default=["*"])

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
