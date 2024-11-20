import json

# Input and output file paths
input_file = "non_multi_turn_examples.json"  # Replace with your JSON file name
output_file = "output.txt"

# Load JSON data
with open(input_file, "r") as infile:
    data = json.load(infile)

# Prepare text content
with open(output_file, "w") as outfile:
    for i, entry in enumerate(data, start=1):
        outfile.write(f"Entry {i}\n")
        outfile.write("=" * 40 + "\n")
        outfile.write(f"Original Instruction:\n{entry.get('original_instruction', 'N/A')}\n\n")
        outfile.write(f"Generated Instruction:\n{entry.get('generated_instruction', 'N/A')}\n\n")
        outfile.write(f"Response:\n{entry.get('response', 'N/A')}\n")
        outfile.write("\n" + "=" * 40 + "\n\n")

print(f"Formatted text file saved to {output_file}")
