from fastapi import APIRouter
from app.translater.translate_service import translate_text
from app.api.models import TextRequest, TranslationResponse

router = APIRouter()

@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TextRequest):
    result = await translate_text(request.text)
    return TranslationResponse(translations=result)
