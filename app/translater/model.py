from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model():
    model_name = "facebook/mbart-large-50-one-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    return model, tokenizer

def translate_to_all_languages(text: str, model, tokenizer):
    supported_languages = ["ar", "de", "en", "es", "fr", "it", "ja", "ko", "pl", "pt", "ru", "zh"]
    translated_texts = {}

    tokenizer.src_lang = "en_XX"
    encoded_input = tokenizer(text, return_tensors="pt")

    for lang in supported_languages:
        model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang])
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated_texts[lang] = translated_text

    return translated_texts
