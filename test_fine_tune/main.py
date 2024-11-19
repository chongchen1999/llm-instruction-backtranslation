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
def preprocess_data(dataset_path="/home/tourist/Generative-AI/llm-instruction-backtranslation/data/seed/seed.jsonl", tokenizer=None):
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

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]

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
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )

    # Set the format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val

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
def fine_tune_lora(model, tokenizer, train_dataset, val_dataset, output_dir="models/lora-tinyllama"):
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
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-4,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # Changed from "steps" to "epoch"
        logging_dir="./logs",
        logging_steps=50,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,  # Added to prevent column removal
    )

    # Use Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model

# Load Fine-Tuned Model
def load_fine_tuned_model(model_path, base_model_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Generate Response
def generate_response(prompt, model, tokenizer):
    input_text = f"Instruction: {prompt}\n\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Function
if __name__ == "__main__":
    # Load base model and tokenizer
    base_model_path = "models/tinyllama-1.1b"
    seed_data_path = "/home/tourist/Generative-AI/llm-instruction-backtranslation/data/seed/seed.jsonl"
    fine_tuned_model_path = "models/lora-tinyllama"

    model, tokenizer = load_model(base_model_path)
    train_dataset, val_dataset = preprocess_data(seed_data_path, tokenizer)

    # Fine-tune with LoRA
    fine_tune_lora(model, tokenizer, train_dataset, val_dataset, fine_tuned_model_path)

    # Load fine-tuned model
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(fine_tuned_model_path, base_model_path)

    # Test fine-tuned model
    test_prompt = "Explain Z-score normalization."
    print(generate_response(test_prompt, fine_tuned_model, fine_tuned_tokenizer))