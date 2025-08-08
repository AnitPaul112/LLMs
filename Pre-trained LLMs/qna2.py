#decoder for question answering
from transformers import pipeline

question = "Who painted the Mona Lisa?"

# Define context that contains the answer
text = """
The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. 
It has been described as the best known, the most visited, the most written about, 
the most sung about, the most parodied work of art in the world. The painting is 
thought to be a portrait of Lisa Gherardini, the wife of Florentine merchant 
Francesco del Giocondo. Leonardo da Vinci painted it between 1503 and 1519.
"""

# Define the appropriate model (GPT-2 is for text generation, not QA)
generator = pipeline(task="text-generation", model="gpt2")

input_text = f"Context: {text}\n\nQuestion: {question}\n\nAnswer:"

output = generator(input_text, max_length=len(input_text) + 50, num_return_sequences=1, pad_token_id=50256)
print("Generated response:")
print(output[0]['generated_text'][len(input_text):].strip())