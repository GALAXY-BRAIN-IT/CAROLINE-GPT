import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .model import CarolineAI

class CarolineTrainer:
    def __init__(self, model: CarolineAI, learning_rate: float = 3e-4):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler()

    def train_step(self, batch: torch.Tensor):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast('cuda'):
            outputs = self.model(inputs)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.contiguous().view(-1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
