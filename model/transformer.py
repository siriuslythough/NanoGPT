import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadedSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    model_dim
        ↓
    4 * model_dim
        ↓
    activation
        ↓
    model_dim
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                model_dim,
                4 * model_dim
            ),

            nn.ReLU(),

            nn.Linear(
                4 * model_dim,
                model_dim
            ),

            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.network(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm decoder Transformer block.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.attention = MultiHeadedSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads
        )

        self.feed_forward = FeedForward(
            model_dim=model_dim,
            dropout=dropout
        )

        self.first_norm = nn.LayerNorm(
            model_dim
        )

        self.second_norm = nn.LayerNorm(
            model_dim
        )

    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:

            # Pre-LN attention + residual
            x = x + self.attention(
                self.first_norm(x)
            )

            # Pre-LN FFN + residual
            x = x + self.feed_forward(
                self.second_norm(x)
            )

            return x
    def forward_cached(
        self,
        x,
        kv_caches=None,
        max_length=None
    ):

        # Pre-LN
        normalized = self.first_norm(x)

        attention_output, updated_caches = (
            self.attention.forward_cached(
                normalized,
                kv_caches=kv_caches,
                max_length=max_length
            )
        )

        # Attention residual
        x = x + attention_output

        # FFN residual
        x = x + self.feed_forward(
            self.second_norm(x)
        )

        return x, updated_caches