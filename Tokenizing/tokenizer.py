from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Load dataset
data = load_dataset("imdb", split="train")
train_data = data.shard(num_shards=4, index=0)

test_data = load_dataset("imdb", split="test")
test_data = test_data.shard(num_shards=4, index=0)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
tokenized_training_data = tokenizer(
    train_data["text"],
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

tokenized_test_data = tokenizer(
    test_data["text"],
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)
print("Training data tokenized:", tokenized_training_data)
print("Test data tokenized:", tokenized_test_data)

# Define tokenize function for mapping
def tokenize_function(text_data):
    return tokenizer(text_data["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64)

# Tokenize in batches
tokenized_in_batches = train_data.map(tokenize_function, batched=True)

# Tokenize row by row
tokenized_by_row = train_data.map(tokenize_function, batched=False) 