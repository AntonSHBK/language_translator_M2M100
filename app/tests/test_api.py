# tests/test_api.py
import sys
import os
import json

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Service is running"}


def test_translate_text():
    request_data = {
        "texts": ["The head of the United Nations says there is no military solution in Syria"],
        "source_lang": "en",
        "target_langs": ["fr", "es", "de"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "translation" in result
    translation = result["translation"]

    assert isinstance(translation, dict)
    for lang in ["fr", "es", "de"]:
        assert lang in translation
        assert isinstance(translation[lang], list)
        assert len(translation[lang]) == 1  # Один текст -> один перевод
        assert isinstance(translation[lang][0], str)
        assert len(translation[lang][0]) > 0


def test_translate_without_source_lang():
    request_data = {
        "texts": ["This is an automatically detected sentence."],
        "target_langs": ["fr", "de"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    translation = result["translation"]

    assert isinstance(translation, dict)
    for lang in ["fr", "de"]:
        assert lang in translation
        assert isinstance(translation[lang], list)
        assert len(translation[lang]) == 1
        assert isinstance(translation[lang][0], str)
        assert len(translation[lang][0]) > 0


def test_translate_multiple_texts():
    request_data = {
        "texts": ["Hello world", "How are you?"],
        "target_langs": ["fr", "de"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    translation = result["translation"]

    for lang in ["fr", "de"]:
        assert lang in translation
        assert isinstance(translation[lang], list)
        assert len(translation[lang]) == 2  # Два текста -> два перевода
        for translated_text in translation[lang]:
            assert isinstance(translated_text, str)
            assert len(translated_text) > 0


def test_translate_invalid_data():
    request_data = {
        "texts": [],  # Пустой список текстов
        "source_lang": "en",
        "target_langs": ["fr"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 422  # Ошибка валидации


def test_translate_missing_parameter():
    request_data = {
        "texts": ["Hello world"],  # Нет параметра target_langs
        "source_lang": "en"
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 422  # Ошибка валидации


def test_translate_invalid_language():
    request_data = {
        "texts": ["Hello world"],
        "source_lang": "en",
        "target_langs": ["xx"]  # Неверный код языка
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    translation = result["translation"]

    assert "xx" in translation
    assert isinstance(translation["xx"], list)
    assert translation["xx"][0] == "Hello world"  # При ошибке возвращается исходный текст
