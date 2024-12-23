import logging
from typing import List, Dict

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect
import torch

# from app.translater import LANGUAGE_MAPPING


class TranslationModel:
    """
    Класс для загрузки модели и перевода текста с использованием модели mBART.

    Атрибуты:
    - model_name (str): Название модели для загрузки.
    - cache_dir (str): Путь к директории для кэширования модели.
    - supported_languages (List[str]): Список поддерживаемых языков.
    """
    
    LANGUAGE_MAPPING = {
        "ar": "ar_AR", "cs": "cs_CZ", "de": "de_DE", "en": "en_XX", "es": "es_XX", "et": "et_EE", "fi": "fi_FI",
        "fr": "fr_XX", "gu": "gu_IN", "hi": "hi_IN", "it": "it_IT", "ja": "ja_XX", "kk": "kk_KZ", "ko": "ko_KR",
        "lt": "lt_LT", "lv": "lv_LV", "my": "my_MM", "ne": "ne_NP", "nl": "nl_XX", "ro": "ro_RO", "ru": "ru_RU",
        "si": "si_LK", "tr": "tr_TR", "vi": "vi_VN", "zh": "zh_CN", "af": "af_ZA", "az": "az_AZ", "bn": "bn_IN",
        "fa": "fa_IR", "he": "he_IL", "hr": "hr_HR", "id": "id_ID", "ka": "ka_GE", "km": "km_KH", "mk": "mk_MK",
        "ml": "ml_IN", "mn": "mn_MN", "mr": "mr_IN", "pl": "pl_PL", "ps": "ps_AF", "pt": "pt_XX", "sv": "sv_SE",
        "sw": "sw_KE", "ta": "ta_IN", "te": "te_IN", "th": "th_TH", "tl": "tl_XX", "uk": "uk_UA", "ur": "ur_PK",
        "xh": "xh_ZA", "gl": "gl_ES", "sl": "sl_SI"
    }

    def __init__(
            self, 
            model_name: str = "facebook/mbart-large-50-many-to-many-mmt",                 
            cache_dir: str = "./cache", 
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        """
        Инициализация модели перевода.

        :param model_name: Название модели (по умолчанию "facebook/mbart-large-50-one-to-many-mmt").
        :param cache_dir: Путь к директории для кэширования модели.
        :param device: Устройство для выполнения модели ("cuda" или "cpu").
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device

        # Загружаем модель и токенизатор
        self.model = MBartForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.tokenizer: MBart50TokenizerFast = MBart50TokenizerFast.from_pretrained(
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
            if detected_lang not in self.LANGUAGE_MAPPING:
                self.logger.warning(f"Язык {detected_lang} не поддерживается.")
                return None  # Возвращаем None, если язык не поддерживается
            return self.LANGUAGE_MAPPING[detected_lang]
        except Exception as e:
            self.logger.error(f"Не удалось определить язык текста: {e}")
            return None  # Возвращаем None в случае ошибки

    def translate_batch(self, texts: List[str], target_langs: List[str]) -> List[Dict[str, str]]:
        """
        Переводит каждый текст из списка `texts` на каждый язык из списка `target_langs`.

        :param texts: Список текстов для перевода.
        :param target_langs: Список целевых языков для перевода каждого текста.
        :return: Список переведенных текстов.
        """
        translated_texts = []

        for text in texts:
            # Определяем исходный язык
            source_lang = self.detect_language(text)

            if source_lang is None:
                self.logger.warning(f"Язык не поддерживается возвращаем default.")
                translated_texts.append({"default": text})
                continue

            # Подготовка текста для перевода
            self.tokenizer.src_lang = source_lang
            model_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated_text = {}
            # Перевод каждого текста на все целевые языки
            for target_lang in target_langs:
                if target_lang not in self.LANGUAGE_MAPPING.values():
                    continue  # Пропускаем неподдерживаемый язык

                generated_tokens = self.model.generate(
                    **model_inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
                )

                # Декодирование переведенного текста
                translated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                translated_text_test = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translated_text[target_lang] = translated_text
                
            translated_texts.append(translated_text)

        return translated_texts
     
if __name__ == "__main__":
    text = "The head of the United Nations says there is no military solution in Syria"
    
    # Инициализация модели
    model = TranslationModel()

    # Перевод текста с английского на несколько языков
    target_langs = ["fr_XX", "es_XX", "de_DE"]  # Переводим на французский, испанский и немецкий
    translated_batch = model.translate_batch([text], target_langs)
    
    for translated in translated_batch:
        print(translated)