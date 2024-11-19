from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"  # or other model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For inference
inputs = tokenizer("Write a story about:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))