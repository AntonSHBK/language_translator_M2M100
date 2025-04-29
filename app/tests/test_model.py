# tests/test_translate.py
import sys
import os

import pytest
from pathlib import Path

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.translate_model import TranslationModelQAT


@pytest.fixture(scope="module")
def translation_model():
    return TranslationModelQAT(
        model_name="michaelfeil/ct2fast-m2m100_418M",
        cache_dir=Path("./app/data/cache_dir"),
        device="cpu"  # Для тестирования используем CPU
    )

def test_model_loading(translation_model):
    assert translation_model.model is not None, "Модель не была загружена"
    assert translation_model.tokenizer is not None, "Токенизатор не был загружен"

def test_translate_single(translation_model):
    text = "The head of the United Nations says there is no military solution in Syria"
    source_lang = "en"
    target_lang = "fr"

    translated_text = translation_model.translate(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang
    )

    assert translated_text is not None
    assert isinstance(translated_text, str)
    assert len(translated_text) > 0  # Переведённый текст должен быть непустым

def test_translate_batch(translation_model):
    texts = [
        "The head of the United Nations says there is no military solution in Syria",
        "Climate change is the greatest challenge of our time."
    ]
    source_lang = "en"
    target_langs = ["fr", "es", "de"]

    translated_texts = translation_model.translate_batch(
        texts=texts,
        target_langs=target_langs,
        source_lang=source_lang
    )

    assert isinstance(translated_texts, dict)
    assert all(lang in translated_texts for lang in target_langs), "Не все целевые языки присутствуют в ответе"

    for lang in target_langs:
        translations = translated_texts[lang]
        assert isinstance(translations, list)
        assert len(translations) == len(texts), "Количество переводов должно совпадать с количеством текстов"
        for translation in translations:
            assert isinstance(translation, str)
            assert len(translation) > 0, "Переведённый текст не должен быть пустым"

def test_is_language_supported(translation_model):
    supported_lang = "en"
    unsupported_lang = "xx"  # Пример языка, который не поддерживается

    assert translation_model.is_language_supported(supported_lang) == True
    assert translation_model.is_language_supported(unsupported_lang) == False

def test_detect_language(translation_model):
    text = "The head of the United Nations says there is no military solution in Syria"
    detected_lang = translation_model.detect_language(text)

    assert detected_lang == "en"
