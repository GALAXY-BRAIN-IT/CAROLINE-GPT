from caroline_ai import CarolineAI, CarolineTokenizer, CarolineTrainer, MODEL_CONFIG
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=2048):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - seq_length, seq_length):
                self.data.append(tokens[i:i+seq_length+1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CarolineAI(MODEL_CONFIG).to(device)
    tokenizer = CarolineTokenizer(MODEL_CONFIG["vocab_size"])
    trainer = CarolineTrainer(model)
    
    texts = ["your training data here..."] * 1000
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    trainer.save_checkpoint("caroline_checkpoint.pt")

if __name__ == "__main__":
    train()
