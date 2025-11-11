import torch.nn as nn
import torch 
import math
import torch.nn.functional as F 



class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model: int, num_heads: int): 
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model 
        self.head_dim = d_model // num_heads 
        self.num_heads = num_heads

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, L, _ = x.shape
        query = self.query(x) # (B, L, d_model)
        key = self.key(x) # (B, L, d_model)
        value = self.value(x) # (B, L, d_model)

        query = query.view(B, L, self.num_heads, self.head_dim) # (B, L, num_heads, head_dim)
        key = key.view(B, L, self.num_heads, self.head_dim) # (B, L, num_heads, head_dim)
        value = value.view(B, L, self.num_heads, self.head_dim) # (B, L, num_heads, head_dim)

        query = query.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        key = key.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        value = value.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, num_heads, L, L)
        attn_scores = F.softmax(attn_scores, dim=-1) # (B, num_heads, L, L)

        out = torch.matmul(attn_scores, value) # (B, num_heads, L, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        return out