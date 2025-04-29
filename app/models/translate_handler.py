# models/translate_handler.py
from typing import List, Dict, Optional
from pathlib import Path

import torch

from app.models.translate_model import TranslationModel, TranslationModelQAT


class TranslateHandler:
    def __init__(
        self, 
        model_name: str = "michaelfeil/ct2fast-m2m100_418M", 
        cache_dir: Path = Path(""), 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Инициализация обработчика.

        :param model_name: Название модели для загрузки.
        :param cache_dir: Директория для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        """
        self.model = TranslationModelQAT(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )

    def handle_translate_request(
        self, 
        texts: List[str], 
        target_langs: List[str],
        source_lang: Optional[str] = None, 
    ) -> Dict[str, List[str]]:
        """
        Обрабатывает запрос на перевод.

        :param texts: Список текстов для перевода.
        :param source_lang: Исходный язык. Если None, язык определяется автоматически.
        :param target_langs: Список целевых языков для перевода.
        :return: Словарь вида {язык: [переводы]}.
        """
        try:
            translated_texts = self.model.translate_batch(
                texts=texts,
                source_lang=source_lang,
                target_langs=target_langs
            )
            return translated_texts

        except Exception as e:
            self.model.logger.error(f"Ошибка при обработке запроса на перевод: {e}")

            return {lang: texts for lang in target_langs}