from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CowFace API"
    app_version: str = "1.0.0"
    debug: bool = False

    seg_model_path: Optional[str] = None
    det_model_path: Optional[str] = None

    default_threshold: float = 0.5
    default_conf: float = 0.25
    device: str = "auto"  # auto | cpu | cuda

    max_image_size_mb: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
