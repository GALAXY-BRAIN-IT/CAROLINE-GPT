import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import Expert

class MoELayer(nn.Module):
    def __init__(self, d_model, ffn_dim, num_experts=16, top_k=4, lora_rank=16):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, ffn_dim, lora_rank) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        gate_logits = self.gate(x_flat)
        weights = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_mask = top_indices == i
            if expert_mask.any():
                expert_out = self.experts[i](x_flat)
                output += expert_out * top_weights[:, i].unsqueeze(-1) * expert_mask.float().unsqueeze(-1)
        
        return output.view(batch_size, seq_len, d_model)
