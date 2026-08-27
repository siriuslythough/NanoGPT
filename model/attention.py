import math

import torch
import torch.nn as nn

from .kv_cache import KVCache


class SingleHeadAttention(nn.Module):

    def __init__(
        self,
        model_dim: int,
        head_dim: int
    ):
        super().__init__()

        self.head_dim = head_dim

        self.key_gen = nn.Linear(
            model_dim,
            head_dim,
            bias=False
        )

        self.query_gen = nn.Linear(
            model_dim,
            head_dim,
            bias=False
        )

        self.value_gen = nn.Linear(
            model_dim,
            head_dim,
            bias=False
        )

    # -----------------------------------------------------
    # Common attention calculation
    # -----------------------------------------------------

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        query_length = q.size(1)
        key_length = k.size(1)

        # Important:
        # In cached decoding key_length can be greater
        # than query_length.
        #
        # Example:
        #
        # cache has 10 tokens
        # q contains 1 new token
        #
        # Q length = 1
        # K length = 11

        offset = key_length - query_length

        mask = torch.tril(
            torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=q.device
            ),
            diagonal=offset
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        return weights @ v

    # -----------------------------------------------------
    # Standard path: training
    # -----------------------------------------------------

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        q = self.query_gen(x)
        k = self.key_gen(x)
        v = self.value_gen(x)

        return self._attention(
            q,
            k,
            v
        )

    # -----------------------------------------------------
    # Cached path: autoregressive inference
    # -----------------------------------------------------

    def forward_cached(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        max_length: int | None = None
    ):

        # ONLY compute Q/K/V for the newly supplied tokens
        q = self.query_gen(x)
        new_k = self.key_gen(x)
        new_v = self.value_gen(x)

        if kv_cache is None:
            kv_cache = KVCache(
                max_length=max_length
            )

        # Old K/V + newly calculated K/V
        full_k, full_v = kv_cache.update(
            new_k,
            new_v
        )

        output = self._attention(
            q,
            full_k,
            full_v
        )

        return output, kv_cache