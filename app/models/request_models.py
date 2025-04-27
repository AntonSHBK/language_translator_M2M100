
from typing import List, Optional

from pydantic import BaseModel, Field, constr, field_validator

class TranslateRequest(BaseModel):
    texts: List[constr(min_length=1)] = Field(
        ...,
        description="Список текстов для перевода (каждый текст должен быть непустым)",
        examples=[["Hello", "Good morning"]],
    )
    target_langs: List[str] = Field(
        ...,
        description="Список целевых языков перевода (ISO-коды, например: 'fr', 'de', 'es')",
        examples=[["fr", "de", "es"]],
    )
    source_lang: Optional[str] = Field(
        None,
        description="Исходный язык текста (например: 'en'). Если не указан, язык будет определён автоматически.",
        examples=["en"],
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Список текстов не может быть пустым.')
        return v

    @field_validator('target_langs')
    @classmethod
    def validate_target_langs(cls, v):
        if not v:
            raise ValueError('Список целевых языков не может быть пустым.')
        return v
