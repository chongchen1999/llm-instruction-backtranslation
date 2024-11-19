from datasets import load_dataset

# Load the dataset
dataset = load_dataset("GAIR/lima")

# Filter out multi-turn examples (keeping only single-turn conversations)
single_turn_dataset = dataset['train'].filter(
    lambda example: len(example['conversations']) == 2
)

# Print 5 examples
for i, example in enumerate(single_turn_dataset.select(range(5)), 1):
    print(f"Example {i}:")
    print("Conversation:", example['conversations'])
    print("Source:", example['source'])
    print("-" * 50)