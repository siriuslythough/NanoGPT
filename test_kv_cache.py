import torch

from model.gpt import GPT


# =========================================================
# Architecture
# =========================================================

VOCAB_SIZE = 512
CONTEXT_LENGTH = 128

MODEL_DIM = 384
NUM_BLOCKS = 6
NUM_HEADS = 6


torch.manual_seed(42)


model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    model_dim=MODEL_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS
)

# Critical: disable dropout
model.eval()


# =========================================================
# Random fake sequence
# =========================================================

tokens = torch.randint(
    0,
    VOCAB_SIZE,
    (1, 32)
)


# =========================================================
# TEST 1
#
# Full prefill should equal ordinary forward
# =========================================================

with torch.no_grad():

    ordinary_logits = model(tokens)

    cached_logits, cache = (
        model.forward_cached(
            tokens,
            cache=None
        )
    )


max_difference = (
    ordinary_logits
    - cached_logits
).abs().max().item()


print(
    f"Prefill max difference: "
    f"{max_difference:.8f}"
)


assert torch.allclose(
    ordinary_logits,
    cached_logits,
    atol=1e-5
)

print(
    "Prefill equivalence passed."
)


# =========================================================
# TEST 2
#
# Incremental decoding should equal recomputing
# the entire sequence.
# =========================================================

prefix_length = 8

prefix = tokens[
    :, :prefix_length
]


with torch.no_grad():

    _, cache = model.forward_cached(
        prefix,
        cache=None
    )

    worst_difference = 0.0

    for position in range(
        prefix_length,
        tokens.size(1)
    ):

        # ONLY one new token supplied
        new_token = tokens[
            :,
            position:position + 1
        ]

        cached_step_logits, cache = (
            model.forward_cached(
                new_token,
                cache=cache
            )
        )

        # Reference:
        # expensive full-prefix recomputation
        full_prefix = tokens[
            :,
            :position + 1
        ]

        reference_logits = model(
            full_prefix
        )[:, -1:, :]

        difference = (
            cached_step_logits
            - reference_logits
        ).abs().max().item()

        worst_difference = max(
            worst_difference,
            difference
        )


print(
    f"Incremental max difference: "
    f"{worst_difference:.8f}"
)


assert worst_difference < 1e-5

print(
    "Incremental KV-cache "
    "equivalence passed."
)


# =========================================================
# Inspect cache
# =========================================================

print(
    f"Final cache length: "
    f"{cache[0][0].length}"
)

print(
    f"Layers cached: "
    f"{len(cache)}"
)

print(
    f"Heads per layer: "
    f"{len(cache[0])}"
)

print(
    f"One K tensor shape: "
    f"{cache[0][0].cache_k.shape}"
)