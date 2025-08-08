from transformers import pipeline
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-de")
text = "Hello, how are you?"
translation = translator(text, clean_up_tokenization=True)
print(translation[0]['translation_text'])