import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, context_length, model_dim, num_heads, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(model_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        token_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
