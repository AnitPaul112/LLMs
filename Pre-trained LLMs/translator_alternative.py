from transformers import pipeline

# Using a different translation model that works well
translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", 
                     src_lang="eng_Latn", tgt_lang="deu_Latn")

text = "Hello, how are you?"
translation = translator(text, max_length=40)
print(f"Original: {text}")
print(f"Translation: {translation[0]['translation_text']}")
