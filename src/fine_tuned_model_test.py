import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_fine_tuned_model(model_path, base_model_path):
    # Detect available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge LoRA weights with base model and move to device
    model = model.merge_and_unload()
    model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def generate_response(prompt, model, tokenizer, device):
    input_text = f"Instruction: {prompt}\n\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate with more controlled parameters
    outputs = model.generate(
        inputs.input_ids, 
        max_length=512, 
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Test prompts
    test_prompts = [
        "Explain Z-score normalization.",
        "What are the key differences between supervised and unsupervised machine learning?",
        "Describe the process of gradient descent in neural networks."
    ]

    # Paths
    base_model_path = "models/tinyllama-1.1b"
    fine_tuned_model_path = "models/lora-tinyllama"

    # Load fine-tuned model with GPU support
    model, tokenizer, device = load_fine_tuned_model(fine_tuned_model_path, base_model_path)

    # Generate responses
    print("Fine-Tuned Model Responses:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(prompt, model, tokenizer, device)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()