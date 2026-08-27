import torch
import torch.nn as nn

from .attention import SingleHeadAttention


class MultiHeadedSelfAttention(nn.Module):
    """
    Runs multiple causal attention heads in parallel,
    concatenates them, then applies an output projection.

    Input:
        (B, T, model_dim)

    Output:
        (B, T, model_dim)
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int
    ):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads"
            )

        head_dim = model_dim // num_heads

        self.att_heads = nn.ModuleList([
            SingleHeadAttention(
                model_dim=model_dim,
                head_dim=head_dim
            )
            for _ in range(num_heads)
        ])

        self.output_proj = nn.Linear(
            model_dim,
            model_dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        head_outputs = [
            head(x)
            for head in self.att_heads
        ]

        # Each head: (B, T, head_dim)
        # Combined:  (B, T, model_dim)
        concatenated = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.output_proj(
            concatenated
        )

        return output
    def forward_cached(
        self,
        x: torch.Tensor,
        kv_caches=None,
        max_length=None
    ):

        if kv_caches is None:
            kv_caches = [
                None
                for _ in self.att_heads
            ]

        if len(kv_caches) != len(self.att_heads):
            raise ValueError(
                "Number of KV caches must match "
                "number of attention heads."
            )

        head_outputs = []
        updated_caches = []

        for head, cache in zip(
            self.att_heads,
            kv_caches
        ):

            output, cache = head.forward_cached(
                x,
                kv_cache=cache,
                max_length=max_length
            )

            head_outputs.append(output)
            updated_caches.append(cache)

        concatenated = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.output_proj(
            concatenated
        )

        return output, updated_caches