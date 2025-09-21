from caroline_ai import CarolineAI, CarolineTokenizer, MODEL_CONFIG
import torch

def generate_text(prompt, max_length=100, temperature=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CarolineAI(MODEL_CONFIG).to(device)
    tokenizer = CarolineTokenizer(MODEL_CONFIG["vocab_size"])
    
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            tokens.append(next_token.item())
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.vocab.get("<|endoftext|>", 0):
                break
    
    return tokenizer.decode(tokens)

print(generate_text("The future of AI is"))
