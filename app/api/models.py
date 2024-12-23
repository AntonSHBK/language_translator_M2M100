from pydantic import BaseModel
from typing import Dict

class TextRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translations: Dict[str, str]  # Словарь с языками и переведенными текстами
