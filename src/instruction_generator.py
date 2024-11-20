import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

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

def generate_instruction(completion, model, tokenizer, device, max_length=1024):
    # Format the completion for instruction generation
    input_text = f"Response: {completion}\n\nInstruction:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate instruction with controlled parameters
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    
    instruction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the initial response from output to get just the instruction
    instruction = instruction[len(input_text):].strip()
    return instruction

def process_lima_dataset(num_samples=150):
    # Load LIMA dataset
    dataset = load_dataset("GAIR/lima")
    
    # Filter out multi-turn examples (keeping only single-turn conversations)
    single_turn_dataset = dataset['train'].filter(
        lambda example: len(example['conversations']) == 2
    )
    
    # Paths for the model
    base_model_path = "models/tinyllama-1.1b"
    fine_tuned_model_path = "models/huggingface/tinyllama-backtranslation-model-Myx"
    
    # Load the backtranslation model
    model, tokenizer, device = load_fine_tuned_model(fine_tuned_model_path, base_model_path)
    
    # Get responses from LIMA dataset (taking only the responses from single-turn conversations)
    conversations = []
    for item in single_turn_dataset.select(range(num_samples)):
        # The second message in conversations is the response
        conversation = item['conversations']
        conversations.append(conversation)
    
    # Generate instructions for each response
    generated_pairs = []
    for conversation in tqdm(conversations, desc="Generating Instructions"):
        try:
            response = conversation[1]
            instruction = generate_instruction(response, model, tokenizer, device)
            # Also store the original instruction for comparison
            original_instruction = conversation[0]
            generated_pairs.append({
                'original_instruction': original_instruction,
                'generated_instruction': instruction,
                'response': response,
            })
        except Exception as e:
            print(f"Error processing response: {e}")
            continue
    
    # Save results
    save_results(generated_pairs)
    
    return generated_pairs

def save_results(generated_pairs):
    """Save the generated instruction-response pairs to a file"""
    import json

    filename = f"dataset/lima/lima_instructions.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(generated_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filename}")

def main():
    print("Starting LIMA dataset instruction generation...")
    generated_pairs = process_lima_dataset(num_samples=250)
    print(f"Successfully generated {len(generated_pairs)} instruction-response pairs")
    
    # Print a few examples
    print("\nExample generations:")
    for i, pair in enumerate(generated_pairs[:3]):
        print(f"\nExample {i+1}:")
        print(f"Original Instruction: {pair['original_instruction']}")
        print(f"Response: {pair['response'][:200]}...")
        print(f"Generated Instruction: {pair['generated_instruction']}")

if __name__ == "__main__":
    main()