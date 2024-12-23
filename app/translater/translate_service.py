from app.translater.model import load_model, translate_to_all_languages

model, tokenizer = load_model()

async def translate_text(text: str):
    return translate_to_all_languages(text, model, tokenizer)
