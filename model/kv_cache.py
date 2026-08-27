import torch


class KVCache:
    """
    Preallocated KV cache for all attention heads
    in one Transformer layer.

    Expected K/V layout:

        (B, H, T, head_dim)
    """

    def __init__(self, max_length: int):

        if max_length is None:
            raise ValueError(
                "max_length must be provided."
            )

        self.max_length = max_length

        self.cache_k = None
        self.cache_v = None

        self.current_length = 0


    @property
    def length(self):
        return self.current_length


    def _initialize(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ):
        """
        new_k/new_v:

            (B, H, T, head_dim)
        """

        if new_k.ndim != 4:
            raise ValueError(
                f"Expected new_k to have 4 dimensions "
                f"(B, H, T, head_dim), "
                f"got shape {tuple(new_k.shape)}."
            )

        if new_v.shape != new_k.shape:
            raise ValueError(
                f"K/V shapes must match. "
                f"K={tuple(new_k.shape)}, "
                f"V={tuple(new_v.shape)}"
            )


        B, H, _, head_dim = new_k.shape


        # IMPORTANT AXIS ORDER:
        #
        # B, H, max_length, head_dim
        #
        # NOT:
        # B, max_length, H, head_dim

        self.cache_k = torch.empty(
            (
                B,
                H,
                self.max_length,
                head_dim
            ),
            dtype=new_k.dtype,
            device=new_k.device
        )


        self.cache_v = torch.empty(
            (
                B,
                H,
                self.max_length,
                head_dim
            ),
            dtype=new_v.dtype,
            device=new_v.device
        )


    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor
    ):
        """
        Add newly calculated K/V values.

        Input:
            new_k/new_v:
                (B, H, T_new, head_dim)

        Return:
            full_k/full_v:
                (B, H, T_cached, head_dim)
        """

        if self.cache_k is None:

            self._initialize(
                new_k,
                new_v
            )


        # -------------------------------------------------
        # Sanity check axis ordering
        # -------------------------------------------------

        if new_k.ndim != 4:

            raise ValueError(
                f"Expected new_k shape "
                f"(B,H,T,D), got "
                f"{tuple(new_k.shape)}."
            )


        B, H, T_new, head_dim = (
            new_k.shape
        )


        expected_B = (
            self.cache_k.size(0)
        )

        expected_H = (
            self.cache_k.size(1)
        )

        expected_head_dim = (
            self.cache_k.size(3)
        )


        if B != expected_B:

            raise ValueError(
                f"Batch mismatch: "
                f"{B} vs {expected_B}"
            )


        if H != expected_H:

            raise ValueError(
                f"Head mismatch: "
                f"{H} vs {expected_H}"
            )


        if head_dim != expected_head_dim:

            raise ValueError(
                f"Head dimension mismatch: "
                f"{head_dim} vs "
                f"{expected_head_dim}"
            )


        # -------------------------------------------------
        # Position to write new tokens
        # -------------------------------------------------

        start = self.current_length
        end = start + T_new


        if end > self.max_length:

            raise ValueError(
                f"KV cache would reach length {end}, "
                f"but max context is "
                f"{self.max_length}."
            )


        # -------------------------------------------------
        # Target:
        #
        # cache_k[:, :, start:end, :]
        #
        # Shape:
        # (B, H, T_new, head_dim)
        #
        # EXACTLY same as new_k.
        # -------------------------------------------------

        target_k = self.cache_k[
            :,
            :,
            start:end,
            :
        ]


        target_v = self.cache_v[
            :,
            :,
            start:end,
            :
        ]


        if target_k.shape != new_k.shape:

            raise RuntimeError(
                "KV cache write shape mismatch:\n"
                f"target K: {tuple(target_k.shape)}\n"
                f"new K:    {tuple(new_k.shape)}\n"
                f"cache:    {tuple(self.cache_k.shape)}"
            )


        target_k.copy_(
            new_k
        )

        target_v.copy_(
            new_v
        )


        self.current_length = end


        # Return only populated portion
        return (
            self.cache_k[
                :,
                :,
                :end,
                :
            ],

            self.cache_v[
                :,
                :,
                :end,
                :
            ]
        )


    def clear(self):

        self.current_length = 0