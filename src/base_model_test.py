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
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the initial prompt from response
    response = response[len(input_text):].strip()
    return response

def main():
    # Test prompts
    test_prompts = [
        "Certainly! Here's a serious police report based off the joke:\n\nIncident Report: Burglary at Local Drugstore\n\nOn the evening of [date], two unknown individuals broke into [drugstore name]. A thorough investigation of the crime scene revealed that the suspects stole all of the prescription medication \"Viagra\". The police department has issued an alert to be on the lookout for these two suspects, who are considered dangerous and potentially armed.\n\nDescription of Suspects:\n\n* Suspect 1: Male, estimated to be between the ages of 25 and 35, approximately 6 feet tall with a medium build. He was wearing a dark hoodie and jeans at the time of the break-in.\n* Suspect 2: Male, estimated to be between the ages of 30 and 40, approximately 5'8\" with a heavy build. He was wearing a light-colored hoodie and sweatpants at the time of the break-in.\n\nAnyone with information regarding the whereabouts of these suspects or this crime is encouraged to contact the local police department immediately. The public is advised to use caution and not approach these individuals if spotted.",
        "Sure! Here are a few tools that you can use to automatically generate a dependency graph for your Python codebase:\n\n1. Pyreverse - Pyreverse is a tool that generates UML-style class diagrams and package dependency graphs from Python code. It is part of the PyDev plugin for Eclipse and can also be used as a standalone tool.\n\n2. Graphviz - Graphviz is a powerful open-source graph visualization tool that can be used to generate a variety of diagrams, including class diagrams and package dependency graphs. You can use Graphviz to visualize your Python code by first generating a dot file using a tool like Pyreverse, and then using Graphviz to generate an image from the dot file.\n\n3. Gephi - Gephi is an open-source network visualization tool that can be used to generate class diagrams and package dependency graphs. Gephi provides a number of layout algorithms to arrange nodes and edges, making it possible to generate clear and readable diagrams from large codebases.\n\n4. PlantUML - PlantUML is a tool that generates UML diagrams from a simple text language. It can be used to generate class diagrams and package dependency graphs, and provides a number of customization options to control the appearance of the diagrams.\n\nThese are just a few examples of the many tools available for generating dependency graphs for Python code. Each tool has its own strengths and weaknesses, so it's best to experiment with a few of them to see which one works best for your codebase.",
    ]

    # Load base model
    model, tokenizer = load_base_model()

    # Generate responses
    print("Base Model Responses:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 100)
        response = generate_response(prompt, model, tokenizer)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()