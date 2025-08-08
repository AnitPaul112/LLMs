from datasets import load_dataset

# Load dataset
train_data = load_dataset("imdb", split="train")

# Split data for training
train_data = data.shard(num_shards=4, index=0)

# Load test data
test_data = load_dataset("imdb", split="test")

# Optionally, shard the data to work with smaller portions
train_data = data.shard(num_shards=4, index=0)