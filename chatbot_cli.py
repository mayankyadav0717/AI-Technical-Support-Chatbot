from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "./merged-tinyllama-chatbot"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
model.eval()

print("Chatbot is ready. Type your message (type 'exit' to quit).\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"[INST] {user_input.strip()} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Bot: {response.split('[/INST]')[-1].strip()}\n")
