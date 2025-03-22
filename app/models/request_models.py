from pydantic import BaseModel, Field, constr
from typing import List, Optional


class TranslateRequest(BaseModel):
    texts: List[constr(min_length=1)] = Field(
        ..., 
        min_length=1,
        description="Список текстов для перевода (не может быть пустым)",
        json_schema_extra={"example": ["Hello", "Good morning"]}
    )
    target_langs: List[str] = Field(
        ..., 
        min_length=1,
        description="Список целевых языков (ISO-коды, например: 'fr', 'de', 'es')",
        json_schema_extra={"example": ["fr", "de", "es"]}
    )
    source_lang: Optional[str] = Field(
        None, 
        description="Исходный язык (например: 'en'). Если не указан, определяется автоматически.",
        json_schema_extra={"example": "en"}
    )


class SingleTranslateRequest(BaseModel):
    text: constr(min_length=1) = Field(
        ..., 
        min_length=1,
        description="Один текст для перевода (не может быть пустым)",
        json_schema_extra={"example": "Good evening"}
    )
    target_lang: str = Field(
        ..., 
        min_length=1,
        description="Целевой язык (ISO-код, например: 'fr')",
        json_schema_extra={"example": "fr"}
    )
    source_lang: Optional[str] = Field(
        None, 
        description="Исходный язык (например: 'en'). Если не указан, определяется автоматически.",
        json_schema_extra={"example": "en"}
    )
