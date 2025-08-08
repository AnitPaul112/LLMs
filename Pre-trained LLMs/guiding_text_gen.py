from transformers import pipeline

# Load the text-generation pipeline
generator = pipeline(task="text-generation", model="distilgpt2")

# Input review and response
review = "This book was great. I enjoyed the plot twist in Chapter 10."
response = "Dear reader, thank you for your review. "

# Combine review and response into a prompt
prompt = f"Book review:\n{review}\n\nBook shop response to the review:\n {response}"

# Generate text with specified max length and stop token
output = generator(prompt, max_length=20, pad_token_id=generator.tokenizer.eos_token_id)

# Print the generated text
print(output[0]["generated_text"])
