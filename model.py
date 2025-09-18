import torch.nn as nn
from .block import TransformerBlock
from .config import MODEL_CONFIG

class CarolineAI(nn.Module):
    def __init__(self, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["n_heads"],
                config["ffn_dim"],
                config["num_experts"],
                config["top_k"],
                config["lora_rank"]
            ) for _ in range(config["n_layers"])
        ])
        self.norm = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids, mask=None, use_cache=False):
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, mask, use_cache)
        x = self.norm(x)
        return self.lm_head(x)
