Отличный шаг — интеграция **Celery** в проект позволит тебе запускать **фоновую очередь задач**, например, **перевод текста вне основного потока FastAPI**, не блокируя API при тяжёлых или долгих переводах. Ниже подробный план внедрения:

---

## 🔧 Что такое Celery и зачем он нужен?

**Celery** — это распределённая очередь задач (асинхронных), которая позволяет:
- запускать задачи в фоновом режиме;
- выполнять их параллельно (на рабочих воркерах);
- повторно использовать ресурсоёмкие задачи (перевод, генерация);
- масштабировать нагрузку при большом потоке запросов.

> В твоём случае: **перевод текста** — отличная кандидатура на отправку в очередь.

---

## ✅ Что понадобится:

| Компонент        | Назначение                                      |
|------------------|-------------------------------------------------|
| **Celery**       | Сама очередь задач                              |
| **Redis**        | Брокер (очередь сообщений между API и воркером) |
| **FastAPI**      | Отправляет задачи Celery                        |
| **Worker**       | Обрабатывает задачи (выполняет `translate`)     |

---

## 🧱 Структура проекта после добавления Celery

```
app/
├── service/
│   └── tasks.py          # <-- Celery задачи (перевод)
├── worker.py             # <-- Точка входа для Celery worker
├── routers/
│   └── translate_router.py  # <-- API теперь вызывает celery.delay()
docker/
├── docker-compose.yml    # <-- Redis, worker, app
```

---

## 📦 Установка зависимостей

```bash
pip install celery[redis] redis
```

---

## 🔌 Шаг 1: Конфиг и запуск Celery

### 📝 `app/service/tasks.py`

```python
from celery import Celery
from app.models.translate_handler import TranslateHandler
from app.config import settings

# Создаём приложение Celery
celery_app = Celery(
    "translation_tasks",
    broker="redis://localhost:6379/0",  # можно взять из settings.REDIS_URL
    backend="redis://localhost:6379/0"
)

# Инициализируем модель 1 раз на worker
translate_handler = TranslateHandler(
    model_name=settings.MODEL_NAME,
    cache_dir=settings.CACHE_DIR,
    device=settings.DEVICE
)

@celery_app.task(name="tasks.translate_batch")
def translate_batch_task(texts, source_lang, target_langs):
    """
    Фоновая задача для перевода текста.
    """
    return translate_handler.handle_translate_request(texts, source_lang, target_langs)
```

---

## 🚀 Шаг 2: Создаём worker

### 📝 `worker.py`

```python
from app.service.tasks import celery_app

# Это всё, что нужно — celery автоматически подгрузит задачи
```

---

## ⚙️ Шаг 3: Вызываем задачу из FastAPI

### 📝 В `translate_router.py`

```python
from app.service.tasks import translate_batch_task

@router.post("/api/translate_async")
async def translate_async(request: TranslateRequest):
    """
    Асинхронный перевод текста (в очередь Celery).
    """
    task = translate_batch_task.delay(
        request.texts,
        request.source_lang,
        request.target_langs
    )
    return {"task_id": task.id}
```

---

## 📥 Шаг 4: Эндпоинт для получения результата

```python
from app.service.tasks import celery_app

@router.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """
    Получение результата фоновой задачи.
    """
    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "pending"}
    elif result.state == "SUCCESS":
        return {"status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    return {"status": result.state}
```

---

## 🐳 Шаг 5: `docker-compose.yml` с Redis + worker

```yaml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A app.service.tasks worker --loglevel=info
    depends_on:
      - redis
```

---

## 🧪 Пример запуска:

```bash
# Запустить redis локально (если без Docker)
docker run -d -p 6379:6379 redis

# Запустить FastAPI
uvicorn app.main:app --reload

# Запустить worker отдельно:
celery -A app.service.tasks worker --loglevel=info
```

---

## 💡 Как это работает?

1. Клиент отправляет POST-запрос `/api/translate_async`.
2. FastAPI кладёт задачу в Redis.
3. Celery worker берёт задачу, обрабатывает её (перевод).
4. Клиент периодически запрашивает `/api/result/{task_id}`.
5. Как только задача завершена — получает результат.

---

## 📚 Хочешь:
- пример с прогрессом задачи (через `meta`);
- поддержку очередей с приоритетами;
- или обработку ошибок и retries?

Скажи — допишу нужное!