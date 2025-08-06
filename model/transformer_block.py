import torch.nn as nn
from model.attention import SelfAttention

class FeedForward(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.ReLU(),
            nn.Linear(4 * model_dim, model_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn = SelfAttention(model_dim, model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
