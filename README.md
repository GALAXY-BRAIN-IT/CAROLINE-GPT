## CarolineGPT A 0.5 — Model Definition

This repository contains the model definition and configuration for CarolineGPT A 0.5 — a transformer-based language model designed for ultra-long context processing, scalable training, and efficient adaptation.

### Architecture Overview

- Model type: Transformer decoder-only
- Version: A 0.5 (pre-release architecture)
- Context length: 131,072 tokens
- Vocabulary size: 102,400 tokens
- Hidden dimension (`d_model`): 8192
- Feedforward dimension (`ffn_dim`): 28,672
- Number of layers (`n_layers`): 48
- Attention heads (`n_heads`): 64
- Mixture-of-Experts (MoE): 16 experts, top-4 routing
- LoRA integration: rank 16, alpha 32.0

### Key Features

- Supports extreme sequence lengths for document-level reasoning and retrieval augmentation
- Mixture-of-Experts (MoE) for dynamic routing and compute-efficient scaling
- LoRA-compatible for low-rank fine-tuning and adaptation
- Designed for distributed training across multi-GPU clusters
- Minimal external dependencies, suitable for custom infrastructure

This configuration is intended for research, pretraining, and inference of large-scale autoregressive models with modular extensibility.
