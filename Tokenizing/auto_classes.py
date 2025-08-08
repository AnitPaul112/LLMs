from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification

# Load model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")