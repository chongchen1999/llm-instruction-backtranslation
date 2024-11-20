import json

def analyze_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # 1. Total number of entries
    total_entries = len(data)
    
    # 2. Count non-multi-turn examples
    non_multi_turn_count = sum(
        1 for entry in data 
        if "Response: " not in str(entry.get('generated_instruction', ''))
    )
    
    # 3. Select 5 non-multi-turn examples
    non_multi_turn_examples = [
        entry for entry in data 
        if "Response:: " not in str(entry.get('generated_instruction', ''))
    ][:20]
    
    # Save 5 non-multi-turn examples to a file
    with open('non_multi_turn_examples.json', 'w') as outfile:
        json.dump(non_multi_turn_examples, outfile, indent=2)
    
    # Print results
    print(f"Total entries: {total_entries}")
    print(f"Non-multi-turn entries: {non_multi_turn_count}")
    print("5 non-multi-turn examples saved to 'non_multi_turn_examples.json'")

# Replace 'your_file.json' with the actual path to your JSON file
analyze_json('data/lima/lima_instructions.json')