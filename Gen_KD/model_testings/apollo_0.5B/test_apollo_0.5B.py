"""
Simple chat with Apollo-0.5B model.
Just load and chat - no complexity.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "FreedomIntelligence/Apollo-0.5B"

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device
)
model.eval()

print("✓ Model loaded!\n")
print("=" * 50)
print("Chat with Apollo 0.5B")
print("Type 'quit' to exit")
print("=" * 50 + "\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break
    
    # Generate response
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the input from the response if it's included
    if response.startswith(user_input):
        response = response[len(user_input):].strip()
    
    print(f"Apollo: {response}\n")
