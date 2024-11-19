import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
from peft import LoraConfig, get_peft_model, PeftModel

# Load tokenizer and model
def load_model(model_path="models/tinyllama-1.1b"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer

# Prepare Dataset
def preprocess_data(dataset_path="dataset/seed/seed.jsonl", tokenizer=None):
    # Validate file path
    import os
    print(dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load JSONL file manually
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Create Dataset object
    dataset = Dataset.from_dict({
        'instruction': [item['instruction'] for item in data],
        'response': [item['response'] for item in data]
    })

    def preprocess_function(examples):
        # Combine instruction and response for each example
        combined_texts = [
            f"Instruction: {instruction}\n\nResponse: {response}\n\n"
            for instruction, response in zip(examples['instruction'], examples['response'])
        ]
        
        # Tokenize all texts
        tokenized = tokenizer(
            combined_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )
        
        # Create labels (shifted input_ids)
        tokenized["labels"] = [
            [-100 if mask == 0 else token for mask, token in zip(attention_mask, input_ids)]
            for attention_mask, input_ids in zip(tokenized["attention_mask"], tokenized["input_ids"])
        ]
        
        return tokenized

    # Apply preprocessing
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing full dataset"
    )

    # Set the format for PyTorch
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

# Custom Trainer for handling DataLoader
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass
        outputs = model(**inputs)
        
        # Get the loss
        loss = outputs.loss
        
        # Return loss and outputs if requested
        return (loss, outputs) if return_outputs else loss

# Fine-Tune with LoRA
def fine_tune_lora(model, tokenizer, dataset, output_dir="models/lora-tinyllama"):
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-4,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Use Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model

def main():
    # Paths
    base_model_path = "models/tinyllama-1.1b"
    seed_data_path = "dataset/seed/seed.jsonl"
    fine_tuned_model_path = "models/lora-tinyllama"

    # Load base model and tokenizer
    model, tokenizer = load_model(base_model_path)

    # Prepare full dataset for training
    train_dataset = preprocess_data(seed_data_path, tokenizer)

    # Fine-tune with LoRA
    fine_tune_lora(model, tokenizer, train_dataset, fine_tuned_model_path)

if __name__ == "__main__":
    main()