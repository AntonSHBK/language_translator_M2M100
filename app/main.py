from fastapi import FastAPI
from app.api.endpoints import router as translate_router

app = FastAPI()

# Подключение маршрутов
app.include_router(translate_router, prefix="/api", tags=["translations"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Translation API"}
