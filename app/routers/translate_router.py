import logging

from fastapi import APIRouter
from app.models.request_models import TranslateRequest
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
    summary="Перевод текста(ов) на несколько языков",
    description="""
Переводит один или несколько текстов (`texts`) на указанные целевые языки (`target_langs`).

Если `source_lang` не указан, исходный язык будет определён автоматически.

**Пример запроса:**
```json
{
  "texts": ["Hello world!", "How are you?"],
  "source_lang": "en",
  "target_langs": ["fr", "de"]
}
```
""",
    response_description="Словарь {язык: [переведённые тексты]}."
)
async def translate_text(request: TranslateRequest):
    translated_texts = translate_handler.handle_translate_request(
        texts=request.texts,
        source_lang=request.source_lang,
        target_langs=request.target_langs
    )
    api_logger.info('Перевод успешно выполнен')
    return {"translation": translated_texts}