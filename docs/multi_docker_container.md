Вот краткая и структурированная памятка для документации проекта:

---

# 📘 Памятка: Объединение контейнеров в одну Docker-сеть

## 🧩 Цель:
Обеспечить взаимодействие между двумя или более контейнерами (например, `translate_service` и `telegram_bot`) внутри одного `docker-compose` проекта.

---

## ✅ 1. Создай общую `network` в `docker-compose.yml`

```yaml
networks:
  app_network:
    driver: bridge
```

---

## ✅ 2. Подключи сервисы к этой сети

```yaml
services:
  translate_service:
    container_name: translate_service
    ...
    networks:
      - app_network

  telegram_bot:
    container_name: telegram_bot
    ...
    networks:
      - app_network
```

---

## ✅ 3. Используй имена сервисов как хосты

Внутри одного контейнера другой доступен по **имени сервиса**, например:

```python
# В telegram_bot (Python):
requests.post("http://translate_service:8000/api/translate", json=...)
```

---

## ✅ 4. Uvicorn должен слушать 0.0.0.0

Убедись, что FastAPI-приложение запускается с параметром:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## ✅ 5. Запусти всё:

```bash
docker-compose up --build
```

---

## ✅ 6. Проверка соединения

```bash
docker exec -it telegram_bot ping translate_service
```

Ожидаемый результат:

```
PING translate_service (172.18.0.3): 56 data bytes
64 bytes from ...
```

---

## 📌 Примечание:
- Использовать `localhost` внутри контейнеров **неправильно** — он указывает на сам контейнер.
- DNS-имена сервисов работают только **в пределах одной сети Docker**.

---

Если хочешь — добавим примеры или рисунок-схему.