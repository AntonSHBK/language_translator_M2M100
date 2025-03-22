import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
import torch

from app.utils.logging import setup_logging

# Определяем базовую директорию проекта
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    # Название модели по умолчанию
    MODEL_NAME: str = "facebook/m2m100_418M"

    # Директория для кэширования модели (относительно BASE_DIR)
    CACHE_DIR: Path = Field(default=BASE_DIR / "data" / "cache_dir")

    # Устройство для выполнения модели (CPU или GPU)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Директория для логов (относительно BASE_DIR)
    LOG_DIR: Path = Field(default=BASE_DIR / "logs")

    # Уровень логирования
    LOG_LEVEL: str = "INFO"

    @field_validator("CACHE_DIR", "LOG_DIR", mode="before")
    @classmethod
    def validate_paths(cls, value: str) -> Path:
        """Автоматически создаёт директории при инициализации."""
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

# Экземпляр настроек
settings = Settings()

# Настройка логирования
setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)