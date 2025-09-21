from caroline_ai import CarolineAI, CarolineTokenizer, MODEL_CONFIG
import torch

model = CarolineAI(MODEL_CONFIG)
tokenizer = CarolineTokenizer(MODEL_CONFIG["vocab_size"])

text = "Hello Caroline AI! How are you today?"
tokens = tokenizer.encode(text)
input_tensor = torch.tensor([tokens])

print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

with torch.no_grad():
    output = model(input_tensor)
    
print(f"Output shape: {output.shape}")
print(f"Next token prediction: {torch.argmax(output[0, -1, :]).item()}")
