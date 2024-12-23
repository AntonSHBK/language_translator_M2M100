import pytest
from app.translater.translate_service import translate_text

def test_translation():
    text = "Hello, world!"
    result = translate_text(text)
    assert "en" in result  # Проверяем, что перевод на английский есть
    assert result["en"] == text  # Проверяем, что перевод совпадает с исходным текстом
