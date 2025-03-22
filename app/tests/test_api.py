# tests/test_api.py
import sys
import os

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app

# Создание клиента FastAPI
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
    assert isinstance(result["translation"], list)
    assert len(result["translation"]) == 1

    first_translation = result["translation"][0]
    assert isinstance(first_translation, dict)
    assert "fr" in first_translation
    assert isinstance(first_translation["fr"], str)
    assert len(first_translation["fr"]) > 0
    
       
def test_translate_without_source_lang():
    request_data = {
        "texts": ["This is an automatically detected sentence."],
        "target_langs": ["fr", "de"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "translation" in result
    assert isinstance(result["translation"], list)
    assert len(result["translation"]) == 1

    translations = result["translation"][0]
    assert "fr" in translations and isinstance(translations["fr"], str)
    assert "de" in translations and isinstance(translations["de"], str)
    assert len(translations["fr"]) > 0
    assert len(translations["de"]) > 0


def test_translate_single_text():
    request_data = {
        "text": "The head of the United Nations says there is no military solution in Syria",
        "source_lang": "en",
        "target_lang": "fr"
    }
    response = client.post("/api/translate_single", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "translation" in result
    assert isinstance(result["translation"], dict)
    assert "fr" in result["translation"]
    assert isinstance(result["translation"]["fr"], str)
    assert len(result["translation"]["fr"]) > 0


def test_translate_invalid_data():
    request_data = {
        "texts": [],
        "source_lang": "en",
        "target_langs": ["fr"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 422



def test_translate_missing_parameter():
    request_data = {
        "texts": ["Hello world"],
        "source_lang": "en"
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 422


def test_translate_invalid_language():
    request_data = {
        "texts": ["Hello world"],
        "source_lang": "en",
        "target_langs": ["xx"]
    }
    response = client.post("/api/translate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "translation" in result
    first_translation = result["translation"][0]
    assert "xx" in first_translation
    assert first_translation["xx"] == "Hello world"
