import math

import torch
import torch.nn as nn

from .kv_cache import KVCache


class MultiHeadedSelfAttention(nn.Module):
    """
    Vectorized multi-head causal self-attention.

    Instead of running each attention head as a separate
    PyTorch module, Q/K/V for all heads are generated in
    three large projections and reshaped to:

        (B, H, T, head_dim)

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
                "model_dim must be divisible by num_heads."
            )


        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = (
            model_dim // num_heads
        )


        # -------------------------------------------------
        # Fused projections
        #
        # OLD:
        #     H separate D -> head_dim layers
        #
        # NEW:
        #     one D -> D layer
        #
        # Parameter count is exactly the same.
        # -------------------------------------------------

        self.qkv_gen = nn.Linear(
            model_dim,
            3 * model_dim,
            bias=False
        )


        self.output_proj = nn.Linear(
            model_dim,
            model_dim,
            bias=False
        )

    def _project_qkv(
        self,
        x: torch.Tensor
    ):

        qkv = self.qkv_gen(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        return q, k, v


    # =====================================================
    # Backwards-compatible checkpoint loading
    # =====================================================

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
    ):
        """
        Makes old checkpoints compatible with the new fused QKV layer.

        Supports:

        1. Original checkpoint:
        att_heads.0.query_gen.weight
        att_heads.1.query_gen.weight
        ...

        2. Intermediate vectorized checkpoint:
        query_gen.weight
        key_gen.weight
        value_gen.weight

        3. New checkpoint:
        qkv_gen.weight
        """

        qkv_key = prefix + "qkv_gen.weight"

        # =====================================================
        # CASE 1:
        # Original per-head checkpoint
        # =====================================================

        old_query_keys = [
            prefix + f"att_heads.{i}.query_gen.weight"
            for i in range(self.num_heads)
        ]

        old_key_keys = [
            prefix + f"att_heads.{i}.key_gen.weight"
            for i in range(self.num_heads)
        ]

        old_value_keys = [
            prefix + f"att_heads.{i}.value_gen.weight"
            for i in range(self.num_heads)
        ]

        all_old_head_keys = (
            old_query_keys
            + old_key_keys
            + old_value_keys
        )

        has_old_head_weights = all(
            key in state_dict
            for key in all_old_head_keys
        )

        if (
            qkv_key not in state_dict
            and has_old_head_weights
        ):

            # Each head weight:
            #     (head_dim, model_dim)
            #
            # Concatenate all heads:
            #     (model_dim, model_dim)

            q_weight = torch.cat(
                [
                    state_dict[key]
                    for key in old_query_keys
                ],
                dim=0
            )

            k_weight = torch.cat(
                [
                    state_dict[key]
                    for key in old_key_keys
                ],
                dim=0
            )

            v_weight = torch.cat(
                [
                    state_dict[key]
                    for key in old_value_keys
                ],
                dim=0
            )

            # Q, K, V together:
            #
            # (D, D)
            # (D, D)
            # (D, D)
            #    ↓
            # (3D, D)

            state_dict[qkv_key] = torch.cat(
                [
                    q_weight,
                    k_weight,
                    v_weight
                ],
                dim=0
            )

            # Remove original per-head keys
            # so strict loading does not complain.
            for key in all_old_head_keys:
                state_dict.pop(
                    key,
                    None
                )

        # =====================================================
        # CASE 2:
        # Intermediate vectorized checkpoint
        #
        # query_gen: (D, D)
        # key_gen:   (D, D)
        # value_gen: (D, D)
        # =====================================================

        query_key = prefix + "query_gen.weight"
        key_key = prefix + "key_gen.weight"
        value_key = prefix + "value_gen.weight"

        has_separate_qkv = all(
            key in state_dict
            for key in [
                query_key,
                key_key,
                value_key
            ]
        )

        if (
            qkv_key not in state_dict
            and has_separate_qkv
        ):

            state_dict[qkv_key] = torch.cat(
                [
                    state_dict[query_key],
                    state_dict[key_key],
                    state_dict[value_key]
                ],
                dim=0
            )

            state_dict.pop(
                query_key,
                None
            )

            state_dict.pop(
                key_key,
                None
            )

            state_dict.pop(
                value_key,
                None
            )

        # =====================================================
        # Let PyTorch load everything normally
        # =====================================================

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )


    # =====================================================
    # Shape helpers
    # =====================================================

    def _split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        (B, T, D)
            ->
        (B, H, T, head_dim)
        """

        B, T, _ = x.shape


        x = x.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )


        x = x.transpose(
            1,
            2
        )


        return x


    def _merge_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        (B, H, T, head_dim)
            ->
        (B, T, D)
        """

        B, H, T, head_dim = x.shape


        x = x.transpose(
            1,
            2
        ).contiguous()


        return x.view(
            B,
            T,
            H * head_dim
        )


    # =====================================================
    # Attention
    # =====================================================

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_dim
        )

        query_length = q.size(-2)
        key_length = k.size(-2)


        # -------------------------------------------------
        # Fast incremental decode
        #
        # q contains exactly ONE newly generated token.
        #
        # All keys in the cache are either:
        #   - past tokens
        #   - the current token
        #
        # There are no future keys, so no causal mask
        # is necessary.
        # -------------------------------------------------

        if query_length == 1:

            weights = torch.softmax(
                scores,
                dim=-1
            )

            return weights @ v


        # -------------------------------------------------
        # Standard causal path
        # -------------------------------------------------

        offset = (
            key_length
            - query_length
        )

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


    # =====================================================
    # Standard path
    # =====================================================

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:


        # One large projection for ALL heads
        q, k, v = self._project_qkv(x)


        attention_output = (
            self._attention(
                q,
                k,
                v
            )
        )


        # Back to:
        #     (B,T,D)

        concatenated = (
            self._merge_heads(
                attention_output
            )
        )


        return self.output_proj(
            concatenated
        )


    # =====================================================
    # Cached path
    # =====================================================

    def forward_cached(
        self,
        x: torch.Tensor,
        kv_caches=None,
        max_length=None
    ):
        """
        kv_caches is kept plural for compatibility with
        your current Transformer/GPT interface.

        Internally, however, there is now ONE cache for
        the whole layer instead of one cache per head.
        """


        # -------------------------------------------------
        # Backward-compatible outer cache structure
        #
        # Existing GPT code likely expects something like:
        #
        #     cache[layer][head]
        #
        # To avoid changing GPT immediately, we return a
        # list containing references to the SAME layer cache.
        #
        # Later we can clean the outer API as well.
        # -------------------------------------------------

        if kv_caches is None:

            kv_cache = None

        elif isinstance(
            kv_caches,
            KVCache
        ):

            kv_cache = kv_caches

        elif isinstance(
            kv_caches,
            (list, tuple)
        ):

            if len(kv_caches) == 0:
                kv_cache = None
            else:
                kv_cache = (
                    kv_caches[0]
                )

        else:

            raise TypeError(
                "Unsupported KV-cache format."
            )


        # -------------------------------------------------
        # Compute Q/K/V only for newly supplied tokens
        # -------------------------------------------------

        q, new_k, new_v = (
            self._project_qkv(x)
        )


        if kv_cache is None:

            kv_cache = KVCache(
                max_length=max_length
            )


        full_k, full_v = (
            kv_cache.update(
                new_k,
                new_v
            )
        )


        attention_output = (
            self._attention(
                q,
                full_k,
                full_v
            )
        )


        concatenated = (
            self._merge_heads(
                attention_output
            )
        )


        output = self.output_proj(
            concatenated
        )


        # -------------------------------------------------
        # Compatibility shim:
        #
        # Pretend there are H caches so old GPT code that
        # accesses cache[layer][0].length still works.
        #
        # Every entry references the SAME layer-level cache.
        # -------------------------------------------------

        updated_caches = [
            kv_cache
            for _ in range(
                self.num_heads
            )
        ]


        return (
            output,
            updated_caches
        )