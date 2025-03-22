# routers/translate_router.py
import logging

from fastapi import APIRouter, HTTPException
from app.models.request_models import TranslateRequest, SingleTranslateRequest
from app.models.translate_handler import TranslateHandler
from app.config import settings

router = APIRouter(tags=["Translation"])

api_logger = logging.getLogger("api")

# Инициализация обработчика перевода
translate_handler = TranslateHandler(
    model_name=settings.MODEL_NAME,
    cache_dir=settings.CACHE_DIR,
    device=settings.DEVICE
)

@router.post(
    "/api/translate",
    summary="Перевод списка текстов",
    description="""
Переводит список текстов (`texts`) на указанные целевые языки (`target_langs`).

Если `source_lang` не указан, исходный язык будет определён автоматически.

**Пример запроса:**
```json
{
  "texts": ["Hello", "Good morning"],
  "target_langs": ["fr", "de", "es"],
  "source_lang": "en"
}
```
""",
    response_description="Список переведённых текстов в формате {язык: перевод}. Один словарь на каждый текст."
)
async def translate_text(request: TranslateRequest):
    try:
        translated_texts = translate_handler.handle_translate_request(
            texts=request.texts,
            source_lang=request.source_lang,
            target_langs=request.target_langs
        )
        api_logger.info('Текст переведён')
        return {"translation": translated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/api/translate_single",
    summary="Перевод одного текста",
    description="""
Переводит один текст (`text`) на указанный целевой язык (`target_lang`).

Если `source_lang` не указан, исходный язык будет определён автоматически.

**Пример запроса:**
```json
{
  "text": "Good evening",
  "target_lang": "fr",
  "source_lang": "en"
}
```
""",
    response_description="Словарь с одним переведённым текстом в формате {язык: перевод}"
)
async def translate_single_text(request: SingleTranslateRequest):
    try:
        translated_text = translate_handler.handle_single_translate_request(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        api_logger.info('Текст переведён')
        return {"translation": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))