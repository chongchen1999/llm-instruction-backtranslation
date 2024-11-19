import torch
import json
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class InstructionGenerator:
    def __init__(self, model_path, lima_dataset_path):
        # Load the backward model
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load LIMA dataset
        self.lima_data = self.load_lima_dataset(lima_dataset_path)
    
    def load_lima_dataset(self, path):
        # Assuming LIMA dataset is in JSON format with completions
        with open(path, 'r') as f:
            lima_data = json.load(f)
        
        # Filter out multi-turn examples (single turn only)
        single_turn_data = [
            item for item in lima_data 
            if len(item.get('conversations', [])) == 2
        ]
        
        return single_turn_data
    
    def generate_instructions(self, num_instructions=5, max_length=200):
        generated_pairs = []
        
        for completion in self.lima_data:
            # Use the backward model to generate instructions
            input_text = completion['conversations'][-1]['value']
            
            inputs = self.tokenizer(input_text, return_tensors='pt', 
                                    max_length=max_length, 
                                    truncation=True)
            
            outputs = self.model.generate(
                inputs.input_ids, 
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            generated_instruction = self.tokenizer.decode(outputs[0], 
                                                          skip_special_tokens=True)
            
            generated_pairs.append({
                'instruction': generated_instruction,
                'response': input_text
            })
            
            if len(generated_pairs) >= num_instructions:
                break
        
        return generated_pairs
    
    def load_self_curation_prompt(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def score_quality(self, instruction, response, curation_prompt):
        # Simulate quality scoring based on the provided prompt
        # This is a placeholder - in a real scenario, you'd use an LLM to evaluate
        
        # Simple heuristics for demonstration
        score = 3  # Default neutral score
        
        # Check instruction relevance
        if len(instruction) < 10:
            return 1
        
        # Check response comprehensiveness
        if len(response) > 100 and 'specific details' in response:
            score += 1
        
        # Check for AI-like tone
        if 'As an AI assistant' in response or 'helpful information' in response:
            score += 1
        
        return min(score, 5)  # Cap at 5
    
    def analyze_instruction_quality(self, generated_pairs, curation_prompt_path):
        curation_prompt = self.load_self_curation_prompt(curation_prompt_path)
        
        quality_results = []
        for pair in generated_pairs:
            score = self.score_quality(
                pair['instruction'], 
                pair['response'], 
                curation_prompt
            )
            
            quality_results.append({
                'instruction': pair['instruction'],
                'response': pair['response'],
                'quality_score': score
            })
        
        return quality_results

# Example usage
generator = InstructionGenerator(
    model_path='path/to/backward/model', 
    lima_dataset_path='path/to/lima_dataset.json'
)

# Generate instructions
generated_pairs = generator.generate_instructions(num_instructions=5)

# Analyze quality
quality_analysis = generator.analyze_instruction_quality(
    generated_pairs, 
    'data/prompt/self_curation.txt'
)

# Print results
for result in quality_analysis:
    print(f"Instruction: {result['instruction']}")
    print(f"Response: {result['response']}")
    print(f"Quality Score: {result['quality_score']}\n")