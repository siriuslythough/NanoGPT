import torch
from typing import Optional, Tuple


class KVCache:
    """
    Stores Key and Value tensors for one attention head.

    Shape:
        K: (batch, cached_seq_len, head_dim)
        V: (batch, cached_seq_len, head_dim)
    """

    def __init__(
        self,
        max_length: Optional[int] = None
    ):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

        self.max_length = max_length

    @property
    def length(self) -> int:
        if self.cache_k is None:
            return 0

        return self.cache_k.size(1)

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if new_k.shape != new_v.shape:
            raise ValueError(
                "K and V must have the same shape."
            )

        new_total_length = (
            self.length + new_k.size(1)
        )

        if (
            self.max_length is not None
            and new_total_length > self.max_length
        ):
            raise ValueError(
                f"KV cache would exceed maximum "
                f"context length {self.max_length}."
            )

        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v

        else:
            self.cache_k = torch.cat(
                [self.cache_k, new_k],
                dim=1
            )

            self.cache_v = torch.cat(
                [self.cache_v, new_v],
                dim=1
            )

        return self.cache_k, self.cache_v

    def clear(self) -> None:
        self.cache_k = None
        self.cache_v = None