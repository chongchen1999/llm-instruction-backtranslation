import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_metric

# Directory containing your models
MODEL_DIR = "./models"

# Test inputs for evaluation
TEST_INPUTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Translate 'Hello' to Spanish."
]

# Expected outputs for evaluation (if available)
EXPECTED_OUTPUTS = [
    "Paris.",
    "It's a theory by Einstein explaining space and time.",
    "Hola."
]

def load_model_and_tokenizer(model_path, device):
    """
    Load a model and tokenizer from the given path and move it to the specified device.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Move model to the selected device (GPU if available)
        model.to(device)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        return None, None

def evaluate_model(model, tokenizer, inputs, expected_outputs=None, device='cpu'):
    """
    Evaluate the model on a list of inputs and optionally compare to expected outputs.
    """
    metric = load_metric("bleu") if expected_outputs else None
    generated_outputs = []

    for i, input_text in enumerate(inputs):
        inputs_encoded = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        
        # Move the input tensors to the same device as the model
        inputs_encoded = {key: value.to(device) for key, value in inputs_encoded.items()}
        
        # Generate output on the selected device
        outputs = model.generate(**inputs_encoded, max_length=50, num_beams=4)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_outputs.append(decoded_output)
        print(f"Input: {input_text}")
        print(f"Output: {decoded_output}")
        
        if expected_outputs:
            print(f"Expected: {expected_outputs[i]}")
            metric.add(predictions=[decoded_output.split()], references=[[expected_outputs[i].split()]])
    
    if metric:
        score = metric.compute()
        print(f"Evaluation Score: {score}")
    
    return generated_outputs

def main():
    """
    Main function to load a specified model, run evaluations, and print results.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a specific model.")
    parser.add_argument("--model", required=True, help="Name of the model to evaluate (subdirectory in models folder).")
    args = parser.parse_args()

    # Check if CUDA is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_path = os.path.join(MODEL_DIR, args.model)
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' does not exist.")
        return

    print(f"Evaluating model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    if model and tokenizer:
        evaluate_model(model, tokenizer, TEST_INPUTS, EXPECTED_OUTPUTS, device)
    else:
        print(f"Skipping {model_path} due to loading issues.")

if __name__ == "__main__":
    main()
