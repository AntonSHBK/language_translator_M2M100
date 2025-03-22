# utils/logging.py   
import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === Общий логгер ===
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Файл
    app_file = RotatingFileHandler(Path(log_dir) / "app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    app_file.setFormatter(formatter)
    root_logger.addHandler(app_file)

    # === API логгер ===
    api_logger = logging.getLogger("api")
    api_logger.setLevel(log_level)
    api_file = RotatingFileHandler(Path(log_dir) / "api.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    api_file.setFormatter(formatter)
    api_logger.addHandler(api_file)
    api_logger.propagate = True 

    # === Model логгер ===
    model_logger = logging.getLogger("model")
    model_logger.setLevel(log_level)
    model_file = RotatingFileHandler(Path(log_dir) / "model.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    model_file.setFormatter(formatter)
    model_logger.addHandler(model_file)
    model_logger.propagate = True
