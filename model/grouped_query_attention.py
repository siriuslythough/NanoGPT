import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        q = self.q_proj(x).view(B,T,self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.num_kv_heads, self.head_dim).transpose(1,2)
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        repeats = self.num_heads//self.num_kv_heads
        k = k.repeat_interleave(repeats, dim = 1)
        v = v.repeat_interleave(repeats, dim = 1)
        # 4. Compute scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2,-1))*(self.head_dim**-0.5)
        mask = torch.tril(torch.ones(T,T, device = x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim = -1)
        out = (weights @ v).transpose(1,2).contiguous().view(B,T,-1)
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        return torch.round(self.output_proj(out), decimals=4)
