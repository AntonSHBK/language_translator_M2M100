# models/translate_handler.py
from typing import List, Dict, Optional
from pathlib import Path

import torch

from app.models.translate_model import TranslationModel


class TranslateHandler:
    """
    Обработчик запросов на перевод.

    Атрибуты:
    - model: Модель для перевода текста.
    """

    def __init__(
        self, 
        model_name: str = "facebook/m2m100_418M", 
        cache_dir: Path = Path(""), 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Инициализация обработчика.

        :param model_name: Название модели для загрузки.
        :param cache_dir: Директория для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        """

        # Инициализация модели
        self.model = TranslationModel(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )

    def handle_translate_request(
        self, 
        texts: List[str], 
        source_lang: Optional[str], 
        target_langs: List[str]
    ) -> List[Dict[str, str]]:
        """
        Обрабатывает запрос на перевод.

        :param texts: Список текстов для перевода.
        :param source_lang: Исходный язык. Если None, язык определяется автоматически.
        :param target_langs: Список целевых языков для перевода.
        :return: Список словарей с переведёнными текстами.
        """
        try:
            # Выполняем пакетный перевод
            translated_texts = self.model.translate_batch(
                texts=texts,
                source_lang=source_lang,
                target_langs=target_langs
            )
            return translated_texts
        except Exception as e:
            # Логируем ошибку и возвращаем пустой результат
            self.model.logger.error(f"Ошибка при обработке запроса на перевод: {e}")
            return [{"error": str(e)} for _ in texts]

    def handle_single_translate_request(
        self, 
        text: str, 
        source_lang: Optional[str], 
        target_lang: str
    ) -> Dict[str, str]:
        """
        Обрабатывает запрос на перевод одного текста.

        :param text: Текст для перевода.
        :param source_lang: Исходный язык. Если None, язык определяется автоматически.
        :param target_lang: Целевой язык.
        :return: Словарь с переведённым текстом.
        """
        try:
            # Выполняем перевод
            translated_text = self.model.translate(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            return {target_lang: translated_text}
        except Exception as e:
            # Логируем ошибку и возвращаем сообщение об ошибке
            self.model.logger.error(f"Ошибка при обработке запроса на перевод: {e}")
            return {"error": str(e)}