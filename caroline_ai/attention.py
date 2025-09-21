import torch
import torch.nn as nn
import math
from .lora import LoRALinear

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, lora_rank=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = LoRALinear(nn.Linear(d_model, d_model, bias=False), rank=lora_rank)
        self.wk = LoRALinear(nn.Linear(d_model, d_model, bias=False), rank=lora_rank)
        self.wv = LoRALinear(nn.Linear(d_model, d_model, bias=False), rank=lora_rank)
        self.wo = LoRALinear(nn.Linear(d_model, d_model, bias=False), rank=lora_rank)
        
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, mask=None, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if use_cache and self.cache_k is not None:
            K = torch.cat([self.cache_k, K], dim=2)
            V = torch.cat([self.cache_v, V], dim=2)
        
        if use_cache:
            self.cache_k = K
            self.cache_v = V
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.wo(output)
