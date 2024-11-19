import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base_model(model_path="models/tinyllama-1.1b"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    input_text = f"Instruction: {prompt}\n\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=512, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Test prompts
    test_prompts = [
        "Explain Z-score normalization.",
        "What are the key differences between supervised and unsupervised machine learning?",
        "Describe the process of gradient descent in neural networks."
    ]

    # Load base model
    model, tokenizer = load_base_model()

    # Generate responses
    print("Base Model Responses:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()