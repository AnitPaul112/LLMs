#encoder for question answering
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

# Define the appropriate model
qa = pipeline(task="question-answering", model="distilbert-base-uncased-distilled-squad")

output = qa(question=question, context=text)
print(output['answer'])