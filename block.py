import torch.nn as nn
from .attention import Attention
from .moe import MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, num_experts, top_k, lora_rank):
        super().__init__()
        self.attention = Attention(d_model, n_heads, lora_rank)
        self.moe = MoELayer(d_model, ffn_dim, num_experts, top_k, lora_rank)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, use_cache=False):
        attn_out = self.attention(self.attn_norm(x), mask, use_cache)
        x = x + attn_out
        ffn_out = self.moe(self.ffn_norm(x))
        x = x + ffn_out
        return x
