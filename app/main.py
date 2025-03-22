# main.py
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.routers import translate_router

from app.routers.translate_router import translate_handler

# Инициализация FastAPI
app = FastAPI(
    title="Translation Service",
    description="Сервис для перевода текста.",
    version="1.0.0"
)

# Подключение роутеров
app.include_router(translate_router.router)

@app.get("/")
def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    """
    if not translate_handler.model.model:  # Проверка, что модель загружена
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {"status": "Service is running"}