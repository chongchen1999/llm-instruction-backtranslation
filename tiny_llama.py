import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path="models/tinyllama-1.1b"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto"  # Automatically choose best device (CPU/GPU)
    )
    return model, tokenizer

model, tokenizer = load_model()

# Function for chat
def generate_response(prompt, max_length=512):
    # Format the prompt according to TinyLlama's chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the initial prompt from response
    response = response[len(chat) - 4:].strip()
    return response

# Interactive chat loop
def chat():
    print("Chat with TinyLlama (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        print("\nAssistant:", generate_response(user_input))

# Run the chat
chat()