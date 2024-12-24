import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_translate_single_text():
    # Пример текста для перевода
    data = {
        "text": ["The head of the United Nations says there is no military solution in Syria"],
        "source_lang": "en",
        "target_langs": ["fr", "es", "de"]
    }
    
    response = client.post("/api/translate", json=data)
    
    # Проверка статуса ответа
    assert response.status_code == 200
    
    # Проверка структуры ответа
    json_response = response.json()
    assert "translations" in json_response
    assert isinstance(json_response["translations"], list)
    
    # Проверка перевода на французский, испанский и немецкий
    assert "fr" in json_response["translations"][0]
    assert "es" in json_response["translations"][0]
    assert "de" in json_response["translations"][0]

    # Проверка, что переводы не пустые
    assert json_response["translations"][0]["fr"] != ""
    assert json_response["translations"][0]["es"] != ""
    assert json_response["translations"][0]["de"] != ""

def test_translate_invalid_language():
    # Пример с некорректным целевым языком
    data = {
        "text": ["The head of the United Nations says there is no military solution in Syria"],
        "source_lang": "en",
        "target_langs": ["invalid_language"]
    }

    response = client.post("/api/translate", json=data)
    
    # Проверка статуса ответа
    assert response.status_code == 200
    
    # Проверка структуры ответа, что получен перевод на "default"
    json_response = response.json()
    assert "translations" in json_response
    assert isinstance(json_response["translations"], list)
    assert "invalid_language" not in json_response["translations"][0]
