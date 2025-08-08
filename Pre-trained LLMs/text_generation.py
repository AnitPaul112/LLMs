from transformers import pipeline
generator=pipeline(task="text-generation", model="distilgpt2")

prompt="Bangladesh is famous for"
output=generator(prompt, max_length=50, pad_token_id=generator.tokenizer.eos_token_id)

print(output[0]['generated_text'])