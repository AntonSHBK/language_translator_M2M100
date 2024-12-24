import logging
from typing import List, Dict
from app.translater.model import TranslationModel

class TranslateService:
    """
    Сервис для перевода текста на несколько языков с использованием модели M2M100.
    """

    def __init__(self, model_name: str = "facebook/m2m100_418M", cache_dir: str = "./app/data/cache"):
        """
        Инициализация сервиса перевода.

        :param model_name: Название модели для загрузки.
        :param cache_dir: Путь к директории для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        """
        self.model = TranslationModel(model_name=model_name, cache_dir=cache_dir)
        self.logger = logging.getLogger(__name__)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, List[str]]:
        """
        Переводит текст на указанный целевой язык.

        :param text: Исходный текст для перевода.
        :param source_lang: Исходный язык.
        :param target_lang: Целевой язык для перевода.
        :return: Словарь с целевым языком и списком переведенных строк.
        """
        try:
            # Переводим один текст на целевой язык
            translated_text = self.model.translate_batch([text], source_lang, [target_lang])[0]
            return {target_lang: translated_text.get(target_lang, ["Ошибка перевода"])}
        except Exception as e:
            self.logger.error(f"Ошибка при переводе текста: {e}")
            return {"error": [str(e)]}

    def translate_batch(self, texts: List[str], source_lang: str, target_langs: List[str]) -> List[Dict[str, List[str]]]:
        """
        Переводит несколько текстов на несколько целевых языков.

        :param texts: Список текстов для перевода.
        :param source_lang: Исходный язык.
        :param target_langs: Список целевых языков для перевода.
        :return: Список словарей с переведенными текстами на целевые языки.
        """
        try:
            # Переводим батч текстов
            translated_batch = self.model.translate_batch(texts, source_lang, target_langs)
            return translated_batch
        except Exception as e:
            self.logger.error(f"Ошибка при переводе батча текстов: {e}")
            return [{"error": [str(e)]}]
