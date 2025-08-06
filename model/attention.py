import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.get_keys = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_queries = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_values = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, x):
        k = self.get_keys(x)
        q = self.get_queries(x)
        v = self.get_values(x)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out
