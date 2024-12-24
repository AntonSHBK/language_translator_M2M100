import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_batch_translate():
    # Пример с несколькими текстами для перевода
    data = {
        "text": ["Hello, world!", "How are you?", "Good morning!"],
        "source_lang": "en",
        "target_langs": ["fr", "es", "de"]
    }

    response = client.post("/api/batch_translate", json=data)
    
    # Проверка статуса ответа
    assert response.status_code == 200
    
    # Проверка структуры ответа
    json_response = response.json()
    assert "translations" in json_response
    assert isinstance(json_response["translations"], list)
    
    # Проверка, что переводы для каждого текста на каждый язык присутствуют
    for translation in json_response["translations"]:
        assert "fr" in translation
        assert "es" in translation
        assert "de" in translation
        assert translation["fr"] != ""
        assert translation["es"] != ""
        assert translation["de"] != ""

def test_batch_translate_invalid_language():
    # Пример с несколькими текстами и неверным языком
    data = {
        "text": ["Hello, world!", "How are you?", "Good morning!"],
        "source_lang": "en",
        "target_langs": ["invalid_language"]
    }

    response = client.post("/api/batch_translate", json=data)
    
    # Проверка статуса ответа
    assert response.status_code == 200
    
    # Проверка структуры ответа
    json_response = response.json()
    assert "translations" in json_response
    assert isinstance(json_response["translations"], list)
    
    # Проверка, что неверный язык не добавляется в перевод
    for translation in json_response["translations"]:
        assert "invalid_language" not in translation
