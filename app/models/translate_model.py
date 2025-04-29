import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub
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
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            device_map="auto",  
        )
        self.tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained(
            self.model_name, 
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            use_fast=True
        )
        self.logger.info(f"Модель {self.model_name} успешно загружена на устройство {self.device}.")

    def detect_language(self, text: str) -> Optional[str]:
        """
        Определяет язык текста.

        :param text: Текст для определения языка.
        :return: Код языка или None, если язык не определён.
        """
        self.logger.info("Определение языка")
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
    
    def generate(
        self,
        model_inputs,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        length_penalty: float = 1.2,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        
    ) -> str:
        """
        Генерация перевода с управляемыми параметрами.

        :param model_inputs: Подготовленные входные тензоры.
        :param target_lang: Целевой язык.
        :param max_length: Максимальная длина вывода.
        :param num_beams: Кол-во лучей (beam search).
        :param repetition_penalty: Штраф за повторы.
        :param temperature: Температура генерации.
        :param top_k: Top-k sampling.
        :param top_p: Top-p (nucleus) sampling.
        :param do_sample: Использовать ли сэмплирование.
        :param no_repeat_ngram_size: Запрет повторов одинаковых фраз.
        :param early_stopping: Раннее завершение beam search.
        :return: Переведённый текст.
        """
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )

        return self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        
    def tokenize(
        self, 
        text: str, 
        source_lang: str,
        truncation: bool = True,
        padding: bool = True,
        max_length: int = 512,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Токенизация текста с настраиваемыми параметрами.

        :param text: Текст для токенизации.
        :param source_lang: Код исходного языка.
        :param truncation: Усекать ли текст, если он слишком длинный.
        :param padding: Добавлять ли паддинги.
        :param max_length: Максимальная длина (в токенах).
        :param return_attention_mask: Возвращать ли attention mask.
        :param add_special_tokens: Добавлять ли специальные токены (<s>, </s>, и т.д.).
        :return: Словарь с токенизированным текстом.
        """
        self.tokenizer.src_lang = source_lang
            
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            add_special_tokens=add_special_tokens
        ).to(self.device)
                
        return tokenized_text
    
    def preprocess_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text_to_blocks(
        self,
        text: str,
        source_lang: str,
        max_tokens: int = 256,
        buffer: int = 8
    ) -> List[str]:
        text = self.preprocess_text(text)
        sentences = re.split(r'(?<=[.!?]) +', text)

        blocks = []
        
        for sentence in sentences:

            tokenized = self.tokenize(
                text=sentence,
                source_lang=source_lang,
                truncation=False,
                padding=False,
                max_length=None,
            )
            length = tokenized["input_ids"].shape[1]

            if length + buffer <= max_tokens:
                blocks.append(sentence)
            else:
                words = sentence.split()
                mid = len(words) // 2

                left_part = " ".join(words[:mid])
                right_part = " ".join(words[mid:])

                left_blocks = self.split_text_to_blocks(
                    text=left_part,
                    source_lang=source_lang,
                    max_tokens=max_tokens,
                    buffer=buffer
                )
                right_blocks = self.split_text_to_blocks(
                    text=right_part,
                    source_lang=source_lang,
                    max_tokens=max_tokens,
                    buffer=buffer
                )

                blocks.extend(left_blocks + right_blocks)

        return blocks
    
    def _translate(
        self,
        text: str, 
        source_lang: Optional[str], 
        target_lang: str,
        **kwargs
    ):  
        output_max_length = kwargs.get("output_max_length", 512)
        num_beams = kwargs.get("num_beams", 5)
        length_penalty = kwargs.get("length_penalty", 1.2)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 50)
        top_p = kwargs.get("top_p", 0.95)
        do_sample = kwargs.get("do_sample", False)
        no_repeat_ngram_size = kwargs.get("no_repeat_ngram_size", 3)
        early_stopping = kwargs.get("early_stopping", True)

        truncation = kwargs.get("truncation", True)
        input_max_length = kwargs.get("input_max_length", 256)
        padding = kwargs.get("padding", True)
        add_special_tokens = kwargs.get("add_special_tokens", True)
        
        model_inputs = self.tokenize(
            text, 
            source_lang, 
            truncation=truncation,
            padding=padding,
            max_length=input_max_length,
            add_special_tokens=add_special_tokens
        )
        
        translated_text = self.generate(
            model_inputs=model_inputs,
            target_lang=target_lang,
            max_length=output_max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty
        )
        # self.logger.info("_translate() END")
        return translated_text
            
    def translate(
        self, 
        text: str, 
        target_lang: str,
        source_lang: Optional[str] = None, 
        **kwargs
    ) -> str:
        """
        Переводит текст с одного языка на другой, автоматически разбивая длинные тексты.

        :param text: Текст для перевода.
        :param source_lang: Исходный язык.
        :param target_lang: Целевой язык.
        :param kwargs: Все параметры токенизации и генерации.
        :return: Переведённый текст.
        """
        input_max_length = kwargs.get("input_max_length", 256)
        buffer_window = kwargs.get("buffer_window", 8)
        
        if source_lang is None:
            detected = self.detect_language(text)
            if detected is None:
                return text
            source_lang = detected

        if not self.is_language_supported(source_lang):
            self.logger.warning(f"Исходный язык {source_lang} не поддерживается.")
            return text

        if not self.is_language_supported(target_lang):
            self.logger.warning(f"Целевой язык {target_lang} не поддерживается.")
            return text      
                
        blocks_texts = self.split_text_to_blocks(
            text=text, 
            source_lang=source_lang,
            max_tokens=input_max_length, 
            buffer=buffer_window
        )

        translations = []
        for block in blocks_texts:
            translated = self._translate(
                text=block,
                source_lang=source_lang,
                target_lang=target_lang,
                **kwargs
            )
            translations.append(translated)

        if len(translations) == 1:
            return translations[0]
        return " ".join(translations)

    def translate_batch(
        self,
        texts: List[str],
        target_langs: List[str],
        source_lang: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Переводит список текстов на несколько целевых языков.

        :param texts: Список текстов для перевода.
        :param target_langs: Список целевых языков.
        :param source_lang: Исходный язык (опционально).
        :param kwargs: Дополнительные параметры генерации/токенизации.
        :return: Словарь вида {язык: [переводы]}.
        """
        # Инициализируем пустой результат для каждого языка
        translations = {lang: [] for lang in target_langs}

        for text in texts:
            for target_lang in target_langs:
                translated_text = self.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    **kwargs
                )
                translations[target_lang].append(translated_text)

        return translations


class TranslationModelQAT(TranslationModel):
    """
    Реализация модели переводчика на основе M2M100 с поддержкой QAT.
    """

    def __init__(
        self, 
        model_name: str = "michaelfeil/ct2fast-m2m100_418M", 
        cache_dir: Path = Path(""), 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(model_name, cache_dir, device, **kwargs)

        # Загрузка модели и токенизатора
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """
        Загружает квантованную модель и токенизатор.
        """
        self.logger.info(f"Загрузка модели {self.model_name}...")

        self.tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained(
            "facebook/m2m100_418M",
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            # use_fast=True
        )

        compute_type = "int8_float16" if self.device == "cuda" else "int8"

        self.model = MultiLingualTranslatorCT2fromHfHub(
            model_name_or_path=self.model_name,
            compute_type=compute_type,
            device=self.device,
            tokenizer=self.tokenizer,
            hub_kwargs={"cache_dir": str(self.cache_dir)} if self.cache_dir else {},
        )

        self.logger.info(f"Модель {self.model_name} успешно загружена на устройство {self.device} с compute_type={compute_type}.")
        
    def _translate(
        self,
        text: List[str],
        source_lang: List[str],
        target_lang: List[str],
        **generation_kwargs
    ) -> str:
            
        result = self.model.generate(
            text=text,
            src_lang=source_lang,
            tgt_lang=target_lang,
            **generation_kwargs
        )
        return result
    
    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Переводит длинный текст, при необходимости разбивает на блоки.
        """

        if source_lang is None:
            detected = self.detect_language(text)
            if detected is None:
                return text
            source_lang = detected

        if not self.is_language_supported(source_lang):
            self.logger.warning(f"Исходный язык {source_lang} не поддерживается.")
            return text

        if not self.is_language_supported(target_lang):
            self.logger.warning(f"Целевой язык {target_lang} не поддерживается.")
            return text

        if "max_input_length" not in generation_kwargs:
            generation_kwargs["max_input_length"] = 300

        input_max_length = generation_kwargs["max_input_length"]
        buffer_window = 10

        blocks_texts = self.split_text_to_blocks(
            text=text,
            source_lang=source_lang,
            max_tokens=input_max_length,
            buffer=buffer_window,
        )

        block_translations = self._translate(
            text=blocks_texts,
            source_lang=[source_lang] * len(blocks_texts),
            target_lang=[target_lang] * len(blocks_texts),
            **generation_kwargs
        )

        if len(block_translations) == 1:
            return block_translations[0]
        return " ".join(block_translations)
        
    # def translate_batch(
    #     self,
    #     texts: List[str],        
    #     target_langs: List[str],
    #     source_lang: Optional[str] = None,
    #     **generation_kwargs
    # ) -> List[Dict[str, str]]:
    #     """
    #     Переводит список текстов на список целевых языков.
    #     Для каждого target_lang делается отдельный батч.
    #     """
    #     if not texts:
    #         return []

    #     # Определяем исходные языки
    #     source_languages = []
    #     for text in texts:
    #         if source_lang is None:
    #             detected_lang = self.detect_language(text)
    #             if not detected_lang:
    #                 self.logger.warning(f"Не удалось определить язык текста: {text}")
    #                 source_languages.append(None)
    #             else:
    #                 source_languages.append(detected_lang)
    #         else:
    #             source_languages.append(source_lang)

    #     # Инициализация словаря результатов
    #     translations_by_lang = {}

    #     # Перебор по целевым языкам
    #     for target_lang in target_langs:
    #         if not self.is_language_supported(target_lang):
    #             self.logger.warning(f"Целевой язык {target_lang} не поддерживается.")
    #             translations_by_lang[target_lang] = texts  # если язык не поддерживается — просто возвращаем оригиналы
    #             continue

    #         batch_texts = []
    #         batch_src_langs = []
    #         valid_indices = []

    #         for idx, (text, src) in enumerate(zip(texts, source_languages)):
    #             if src is None or not self.is_language_supported(src):
    #                 batch_texts.append(text)  # fallback на оригинал
    #                 batch_src_langs.append("und")  # special case — undefined
    #             else:
    #                 batch_texts.append(text)
    #                 batch_src_langs.append(src)

    #             valid_indices.append(idx)

    #         try:
    #             results = self.model.generate(
    #                 text=batch_texts,
    #                 src_lang=batch_src_langs,
    #                 tgt_lang=[target_lang] * len(batch_texts),
    #                 **generation_kwargs
    #             )
    #         except Exception as e:
    #             self.logger.error(f"Ошибка при генерации перевода на {target_lang}: {e}")
    #             # Если ошибка — оригинальные тексты
    #             translations_by_lang[target_lang] = texts
    #             continue

    #         translations_by_lang[target_lang] = results

    #     return translations_by_lang