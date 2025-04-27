import requests
import json
import time
import os

# Путь к JSON-файлу
json_path = os.path.join(os.path.dirname(__file__), "long_text.json")

# Загрузка данных
with open(json_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

# URL API
url = "http://127.0.0.1:8000/api/translate"

print("📤 Отправка запроса к API...")

start_time = time.perf_counter()
response = requests.post(url, json=payload)
end_time = time.perf_counter()

duration = end_time - start_time

# Обработка результата
if response.status_code == 200:
    data = response.json()
    print("✅ Успешный ответ от API.\n")

    translations = data.get("translation", {})

    for lang, translated_texts in translations.items():
        print(f"🌍 Язык: {lang}")
        for idx, translated_text in enumerate(translated_texts):
            print(f"  🔹 Текст {idx + 1} ({len(translated_text)} символов)")
            print(f"    → {translated_text}")
        print("-" * 60)

else:
    print(f"❌ Ошибка: статус {response.status_code}")
    print(response.text)

print(f"⏱ Время выполнения: {duration:.2f} секунд")
