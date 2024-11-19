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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the initial prompt from response
    response = response[len(input_text):].strip()
    return response

def main():
    # Test prompts
    test_prompts = [
        "The car is parked in the garage.",
        "Certainly! Here's a serious police report based off the joke:\n\nIncident Report: Burglary at Local Drugstore\n\nOn the evening of [date], two unknown individuals broke into [drugstore name]. A thorough investigation of the crime scene revealed that the suspects stole all of the prescription medication \"Viagra\". The police department has issued an alert to be on the lookout for these two suspects, who are considered dangerous and potentially armed.\n\nDescription of Suspects:\n\n* Suspect 1: Male, estimated to be between the ages of 25 and 35, approximately 6 feet tall with a medium build. He was wearing a dark hoodie and jeans at the time of the break-in.\n* Suspect 2: Male, estimated to be between the ages of 30 and 40, approximately 5'8\" with a heavy build. He was wearing a light-colored hoodie and sweatpants at the time of the break-in.\n\nAnyone with information regarding the whereabouts of these suspects or this crime is encouraged to contact the local police department immediately. The public is advised to use caution and not approach these individuals if spotted.",
        "Sure! Here are a few tools that you can use to automatically generate a dependency graph for your Python codebase:\n\n1. Pyreverse - Pyreverse is a tool that generates UML-style class diagrams and package dependency graphs from Python code. It is part of the PyDev plugin for Eclipse and can also be used as a standalone tool.\n\n2. Graphviz - Graphviz is a powerful open-source graph visualization tool that can be used to generate a variety of diagrams, including class diagrams and package dependency graphs. You can use Graphviz to visualize your Python code by first generating a dot file using a tool like Pyreverse, and then using Graphviz to generate an image from the dot file.\n\n3. Gephi - Gephi is an open-source network visualization tool that can be used to generate class diagrams and package dependency graphs. Gephi provides a number of layout algorithms to arrange nodes and edges, making it possible to generate clear and readable diagrams from large codebases.\n\n4. PlantUML - PlantUML is a tool that generates UML diagrams from a simple text language. It can be used to generate class diagrams and package dependency graphs, and provides a number of customization options to control the appearance of the diagrams.\n\nThese are just a few examples of the many tools available for generating dependency graphs for Python code. Each tool has its own strengths and weaknesses, so it's best to experiment with a few of them to see which one works best for your codebase.",
        "Here are some tips to make your garden more drought resistant: \n\n1. Mix compost into your plant soil. This increases the soil’s capacity to hold water and prevent water pooling and root rot as well. \n\n2. Mulch your plantings. A 5 to 8cm layer of organic mulch shades the soil and slows evaporation. \n\n3. Choose plants that tolerate drought and a wide range of temperatures. Plants that are native to particular regions require less moisture and can help make life easier for stressed wildlife\n\n4. Minimize and drought-proof your container plantings, as these requires more water than ground plants. \n\n5. Cut down on mowing and fertilizing. Mow during the coolest part of the day and leave the clippings on the ground as compost.\n\n6. Reduce your lawn as it consumes a large amount of water and consider drought resistant landscaping instead.\n\n7. Use conserving watering equipment. Watering cans, soaker hoses and drip systems minimize evaporation by sending water directly to the soil instead of spraying it into the air. \n\n8. Water your plants early in the morning, since water doesn’t evaporate as readily in cooler morning temperatures as it does in the heat of the afternoon. You can do it in the evenings, too, but water sitting on leaves overnight may cause fungal diseases.\n\n9. Water less frequently but deeply. Frequent, shallow watering leads to weak, shallow-rooted plants; Less frequent, thorough watering encourages roots to grow deeper.\n\n10. Put off major planting projects until after severe droughts, since newly established plants tend to require a lot of irrigation.",
        "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides.\n\nTo understand the logic behind the Pythagorean theorem, let's consider a right triangle ABC, where C is the right angle and AB is the hypotenuse. We can draw a square on each of the three sides of the triangle, as shown below:\n\n\n          A ________ C\n           |         / \n           |        /   \n           |       /    \n           |      /     \n           |     /      \n           |    /       \n           |   /        \n           |  /         \n           | /          \n          B/           \n\nLet a, b, and c be the lengths of sides AB, BC, and AC, respectively. We can label the areas of the three squares as follows:\n\nArea of square on side AB = a^2\nArea of square on side BC = b^2\nArea of square on side AC = c^2\n\nSince the triangle is right-angled, we know that the two smaller squares on sides AB and BC together form a larger square on side AC. Therefore, the area of the square on side AC is equal to the sum of the areas of the squares on sides AB and BC:\n\na^2 + b^2 = c^2\n\nThis is the Pythagorean theorem in its most basic form. In other words, the Pythagorean theorem is a geometric expression of the relationship between the sides of a right triangle.",
        "The AK-47 is a selective fire, gas operated 7.62x39mm assault rifle. It was the main assault rifle of Eastern Bloc countries during the Cold War and is still used by many militaries today. It is widely regarded as one of the most reliable and recognized weapons in the world. The AK-47 is known for its rugged reliability and its ability to function in a wide range of environments.",
    ]

    # Paths
    base_model_path = "models/tinyllama-1.1b"
    fine_tuned_model_path = "models/huggingface/tinyllama-backtranslation-model-Myx"

    # Load fine-tuned model with GPU support
    model, tokenizer, device = load_fine_tuned_model(fine_tuned_model_path, base_model_path)

    # Generate responses
    print("Fine-Tuned Model Responses:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 100)
        response = generate_response(prompt, model, tokenizer, device)
        print(f"Response---: {response}")
        print("*" * 100)

if __name__ == "__main__":
    main()