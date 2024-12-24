import logging
from typing import List, Dict

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
import torch

class TranslationModel:
    """
    Класс для загрузки модели и перевода текста с использованием модели mBART.

    Атрибуты:
    - model_name (str): Название модели для загрузки.
    - cache_dir (str): Путь к директории для кэширования модели.
    - supported_languages (List[str]): Список поддерживаемых языков.
    """
    
    SUPPORTED_LANGUAGES = [
        "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", 
        "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", 
        "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", 
        "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", 
        "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", 
        "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"
    ]

    def __init__(
            self, 
            model_name: str = "facebook/m2m100_418M",                 
            cache_dir: str = "./cache", 
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        """
        Инициализация модели перевода.

        :param model_name: Название модели (по умолчанию "facebook/m2m100_418M").
        :param cache_dir: Путь к директории для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device

        # Загружаем модель и токенизатор
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.tokenizer: M2M100Tokenizer = M2M100Tokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir)
        
        # Настроим логирование
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_language(self, text: str) -> str:
        """
        Определяет язык текста с использованием библиотеки langdetect.

        :param text: Текст для определения языка.
        :return: Код языка для модели mBART.
        """
        try:
            detected_lang = detect(text)
            if detected_lang not in self.SUPPORTED_LANGUAGES:
                # self.logger.warning(f"Язык {detected_lang} не поддерживается.")
                return None  # Возвращаем None, если язык не поддерживается
            return detected_lang
        except Exception as e:
            # self.logger.error(f"Не удалось определить язык текста: {e}")
            return None  # Возвращаем None в случае ошибки

    def translate_batch(
            self,
            texts: List[str], 
            source_lang: str, 
            target_langs: List[str],
            use_langdetect: bool = True
        ) -> List[Dict[str, str]]:
        """
        Переводит каждый текст из списка `texts` на каждый язык из списка `target_langs`.

        :param texts: Список текстов для перевода.
        :param target_langs: Список целевых языков для перевода каждого текста.
        :return: Список переведенных текстов.
        """
        translated_texts = []

        for text in texts:
            if use_langdetect:
                source_language = self.detect_language(text)

                if source_language is None:
                    source_lang = source_lang
            else:
                source_language = source_lang
                
            if source_language not in self.SUPPORTED_LANGUAGES:                    
                self.logger.warning(f"Язык {source_language} (исходный) не поддерживается.")
                translated_texts.append({"default": text})
                continue
                
            self.tokenizer.src_lang = source_language
            model_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            # model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            translated_text = {}

            for target_lang in target_langs:
                if target_lang not in self.SUPPORTED_LANGUAGES:
                    self.logger.warning(f"Язык {source_language} (целевой) не поддерживается.")
                    translated_text[target_lang] = text
                    continue

                generated_tokens = self.model.generate(
                    **model_inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
                )

                # Декодирование переведенного текста
                translated_text[target_lang] = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated_texts.append(translated_text)

        return translated_texts
     
if __name__ == "__main__":
    text = "The head of the United Nations says there is no military solution in Syria"
    
    # Инициализация модели
    model = TranslationModel()

    # Перевод текста с английского на несколько языков
    target_langs = ["fr", "es", "de"]  # Переводим на французский, испанский и немецкий
    translated_batch = model.translate_batch([text], target_langs)
    
    for translated in translated_batch:
        print(translated)
