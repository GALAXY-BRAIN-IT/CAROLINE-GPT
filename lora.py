import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora_a = nn.Parameter(torch.randn(base_layer.in_features, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, base_layer.out_features))
        self.scaling = alpha / rank
        self.base_layer.requires_grad_(False)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scaling
        return base_out + lora_out
