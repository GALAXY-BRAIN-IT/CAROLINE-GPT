import torch.nn as nn
from .lora import LoRALinear

class Expert(nn.Module):
    def __init__(self, d_model, ffn_dim, lora_rank=16):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)
        
        self.w1_lora = LoRALinear(self.w1, rank=lora_rank)
        self.w2_lora = LoRALinear(self.w2, rank=lora_rank)
        self.w3_lora = LoRALinear(self.w3, rank=lora_rank)
        
        self.act = nn.SiLU()

    def forward(self, x):
        return self.w2_lora(self.act(self.w1_lora(x)) * self.w3_lora(x))
