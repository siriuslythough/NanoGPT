import torch
import torch.nn as nn

from .transformer import TransformerBlock


class GPT(nn.Module):
    """
    Decoder-only Transformer language model.

    token IDs
        ↓
    token embeddings
        +
    positional embeddings
        ↓
    Transformer blocks
        ↓
    final LayerNorm
        ↓
    vocabulary projection
        ↓
    next-token logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dim = model_dim

        # Token embedding table
        self.token_embedding = nn.Embedding(
            vocab_size,
            model_dim
        )

        # Learned positional embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            model_dim
        )

        # Stack of Transformer decoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(
            model_dim
        )

        # Convert hidden representation to vocab logits
        self.lm_head = nn.Linear(
            model_dim,
            vocab_size
        )

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:

        # input_ids: (B, T)

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # (B, T, D)
        token_embeddings = self.token_embedding(
            input_ids
        )

        # (T, D)
        position_embeddings = self.position_embedding(
            positions
        )

        # Broadcasting gives (B, T, D)
        x = (
            token_embeddings
            + position_embeddings
        )

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)

        # (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits
    def forward_cached(
        self,
        input_ids: torch.Tensor,
        cache=None
    ):

        batch_size, sequence_length = (
            input_ids.shape
        )

        # ---------------------------------------------
        # Find how many tokens already exist in cache
        # ---------------------------------------------

        if (
            cache is None
            or len(cache) == 0
            or cache[0] is None
            or len(cache[0]) == 0
            or cache[0][0] is None
        ):
            past_length = 0

        else:
            past_length = cache[0][0].length

        total_length = (
            past_length
            + sequence_length
        )

        if total_length > self.context_length:
            raise ValueError(
                f"Cached sequence length "
                f"{total_length} exceeds context "
                f"length {self.context_length}."
            )

        # ---------------------------------------------
        # Correct absolute positions
        # ---------------------------------------------

        positions = torch.arange(
            past_length,
            total_length,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        # ---------------------------------------------
        # Initialize layer cache slots
        # ---------------------------------------------

        if cache is None:
            cache = [
                None
                for _ in self.transformer_blocks
            ]

        updated_cache = []

        # ---------------------------------------------
        # Cached Transformer blocks
        # ---------------------------------------------

        for block, layer_cache in zip(
            self.transformer_blocks,
            cache
        ):

            x, new_layer_cache = (
                block.forward_cached(
                    x,
                    kv_caches=layer_cache,
                    max_length=self.context_length
                )
            )

            updated_cache.append(
                new_layer_cache
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, updated_cache