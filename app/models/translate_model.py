# models/translate_model.py
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, LangDetectException

from app.models.base_model import BaseTranslationModel


class TranslationModel(BaseTranslationModel):
    """
    Реализация модели переводчика на основе M2M100.
    """

    def __init__(
        self, 
        model_name: str = "facebook/m2m100_418M", 
        cache_dir: Path = Path(""), 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_name, cache_dir, device, **kwargs)

        # Список поддерживаемых языков
        self.supported_languages = [
            "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", 
            "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", 
            "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", 
            "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", 
            "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", 
            "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", 
            "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", 
            "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", 
            "yi", "yo", "zh", "zu"
        ]

        # Загрузка модели и токенизатора
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Загружает модель и токенизатор.
        """
        self.logger.info(f"Загрузка модели {self.model_name}...")
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_name, 
            cache_dir=str(self.cache_dir) if self.cache_dir else None
        ).to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            self.model_name, 
            cache_dir=str(self.cache_dir) if self.cache_dir else None
        )
        self.logger.info(f"Модель {self.model_name} успешно загружена на устройство {self.device}.")

    def detect_language(self, text: str) -> Optional[str]:
        """
        Определяет язык текста.

        :param text: Текст для определения языка.
        :return: Код языка или None, если язык не определён.
        """
        try:
            detected_lang = detect(text)
            if detected_lang in self.supported_languages:
                return detected_lang
            self.logger.warning(f"Язык {detected_lang} не поддерживается.")
            return None
        except LangDetectException as e:
            self.logger.error(f"Ошибка при определении языка: {e}")
            return None
    
    def is_language_supported(self, language: str) -> bool:
        """
        Проверяет, поддерживается ли язык моделью.

        :param language: Код языка (например, "en", "fr").
        :return: True, если язык поддерживается, иначе False.
        """
        if not language:
            self.logger.warning("Пустой код языка.")
            return False

        if language not in self.supported_languages:
            return False

        return True

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Переводит текст с одного языка на другой.

        :param text: Текст для перевода.
        :param source_lang: Исходный язык.
        :param target_lang: Целевой язык.
        :return: Переведённый текст.
        """
        if not self.is_language_supported(source_lang):
            self.logger.warning(f"Исходный язык {source_lang} не поддерживается.")
            return text

        if not self.is_language_supported(target_lang):
            self.logger.warning(f"Целевой язык {target_lang} не поддерживается.")
            return text

        # Устанавливаем исходный язык для токенизатора
        self.tokenizer.src_lang = source_lang

        # Токенизация текста
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Генерация перевода
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )

        # Декодирование переведённого текста
        translated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return translated_text

    def translate_batch(
        self,
        texts: List[str], 
        source_lang: Optional[str], 
        target_langs: List[str]
    ) -> List[Dict[str, str]]:
        """
        Переводит каждый текст из списка `texts` на каждый язык из списка `target_langs`.

        :param texts: Список текстов для перевода.
        :param source_lang: Исходный язык. Если None, язык определяется автоматически.
        :param target_langs: Список целевых языков для перевода каждого текста.
        :return: Список словарей с переведёнными текстами для каждого целевого языка.
        """
        translated_texts = []

        for text in texts:
            # Определяем исходный язык, если он не указан
            if source_lang is None:
                source_language = self.detect_language(text)
                if source_language is None:
                    self.logger.warning(f"Не удалось определить язык текста: {text}")
                    translated_texts.append({"default": text})  # Возвращаем исходный текст
                    continue
            else:
                source_language = source_lang

            # Проверяем, поддерживается ли исходный язык
            if not self.is_language_supported(source_language):
                self.logger.warning(f"Язык {source_language} (исходный) не поддерживается.")
                translated_texts.append({"default": text})
                continue

            # Устанавливаем исходный язык для токенизатора
            self.tokenizer.src_lang = source_language

            # Токенизация текста
            model_inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)

            # Переводим текст на каждый целевой язык
            translated_text = {}
            for target_lang in target_langs:
                # Проверяем, поддерживается ли целевой язык
                if not self.is_language_supported(target_lang):
                    self.logger.warning(f"Язык {target_lang} (целевой) не поддерживается.")
                    translated_text[target_lang] = text  # Возвращаем исходный текст
                    continue

                # Генерация перевода
                generated_tokens = self.model.generate(
                    **model_inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
                )

                # Декодирование переведённого текста
                translated_text[target_lang] = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]  # Берем первый элемент, так как batch_decode возвращает список

            # Добавляем результат перевода в общий список
            translated_texts.append(translated_text)

        return translated_texts
