# models/base_model.py
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional


class BaseTranslationModel(ABC):
    """
    Базовый класс для модели переводчика.

    Атрибуты:
    - supported_languages (List[str]): Список поддерживаемых языков.
    - logger (logging.Logger): Логгер для класса.
    """

    def __init__(
        self, 
        model_name: str, 
        cache_dir: Path = Path("./cache_dir"), 
        device: str = "cpu",
        **kwargs
    ):
        """
        Инициализация базового класса.

        :param model_name: Название модели для загрузки.
        :param cache_dir: Директория для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        :param kwargs: Дополнительные параметры.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.params_dict = kwargs  # Дополнительные параметры
        self.supported_languages: List[str] = []
        
        # Используем логгер из utils
        self.logger = logging.getLogger("model")


    @abstractmethod
    def load_model(self):
        """
        Загружает модель и токенизатор.
        """
        pass

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Переводит текст с одного языка на другой.

        :param text: Текст для перевода.
        :param source_lang: Исходный язык.
        :param target_lang: Целевой язык.
        :return: Переведённый текст.
        """
        pass

    @abstractmethod
    def translate_batch(self, texts: List[str], source_lang: str, target_langs: List[str]) -> List[Dict[str, str]]:
        """
        Переводит список текстов на несколько языков.

        :param texts: Список текстов для перевода.
        :param source_lang: Исходный язык.
        :param target_langs: Список целевых языков.
        :return: Список словарей с переведёнными текстами.
        """
        pass

    def is_language_supported(self, language: str) -> bool:
        """
        Проверяет, поддерживается ли язык.

        :param language: Код языка.
        :return: True, если язык поддерживается, иначе False.
        """
        return language in self.supported_languages

    def detect_language(self, text: str) -> Optional[str]:
        """
        Определяет язык текста (опционально).

        :param text: Текст для определения языка.
        :return: Код языка или None, если язык не определён.
        """
        self.logger.warning("Метод detect_language не реализован.")
        return None