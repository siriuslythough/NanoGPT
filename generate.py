import argparse

import torch
import torch.nn.functional as F

from data.tokenizer import ByteBPETokenizer
from model.gpt import GPT


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/best.pt"
)

parser.add_argument(
    "--tokenizer",
    type=str,
    default="tokenizer.json"
)

parser.add_argument(
    "--prompt",
    type=str,
    default="ROMEO:"
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=80
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.8
)

parser.add_argument(
    "--top-k",
    type=int,
    default=40
)

parser.add_argument(
    "--seed",
    type=int,
    default=None
)

args = parser.parse_args()


# =========================================================
# Device
# =========================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# =========================================================
# Optional reproducibility
# =========================================================

if args.seed is not None:
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


# =========================================================
# Load tokenizer
# =========================================================

tokenizer = ByteBPETokenizer.load(
    args.tokenizer
)


# =========================================================
# Load checkpoint
# =========================================================

checkpoint = torch.load(
    args.checkpoint,
    map_location=device
)

config = checkpoint["config"]


# Safety check
if tokenizer.vocab_size != config["vocab_size"]:
    raise ValueError(
        f"Tokenizer vocabulary ({tokenizer.vocab_size}) "
        f"does not match checkpoint vocabulary "
        f"({config['vocab_size']})."
    )


# =========================================================
# Reconstruct model
# =========================================================

model = GPT(
    **config
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print(
    f"Loaded checkpoint from step "
    f"{checkpoint['step']}"
)


# =========================================================
# Encode prompt
# =========================================================

prompt_ids = tokenizer.encode(
    args.prompt
)

if len(prompt_ids) == 0:
    raise ValueError(
        "Prompt must produce at least one token."
    )


context_length = config[
    "context_length"
]


if len(prompt_ids) >= context_length:
    raise ValueError(
        f"Prompt is {len(prompt_ids)} tokens long, "
        f"but context length is only {context_length}."
    )


input_ids = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)


# =========================================================
# Sampling helper
# =========================================================

def sample_next_token(
    logits,
    temperature,
    top_k
):

    if temperature <= 0:
        # Greedy decoding
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    logits = logits / temperature

    if (
        top_k is not None
        and top_k > 0
    ):

        k = min(
            top_k,
            logits.size(-1)
        )

        values, _ = torch.topk(
            logits,
            k
        )

        cutoff = values[:, -1:]

        logits = logits.masked_fill(
            logits < cutoff,
            float("-inf")
        )

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    return torch.multinomial(
        probabilities,
        num_samples=1
    )


# =========================================================
# KV-cached generation
# =========================================================

generated_ids = prompt_ids.copy()

with torch.no_grad():

    # -----------------------------------------------------
    # PREFILL
    #
    # Process whole prompt once and build KV cache.
    # -----------------------------------------------------

    logits, cache = model.forward_cached(
        input_ids,
        cache=None
    )

    # Prediction after final prompt token
    next_logits = logits[:, -1, :]


    # -----------------------------------------------------
    # Decode autoregressively
    # -----------------------------------------------------

    max_possible_new_tokens = (
        context_length
        - len(prompt_ids)
    )

    tokens_to_generate = min(
        args.max_new_tokens,
        max_possible_new_tokens
    )

    if (
        args.max_new_tokens
        > max_possible_new_tokens
    ):
        print(
            f"Requested {args.max_new_tokens} new tokens, "
            f"but only {max_possible_new_tokens} fit "
            f"inside context length {context_length}."
        )


    for _ in range(
        tokens_to_generate
    ):

        next_token = sample_next_token(
            next_logits,
            temperature=args.temperature,
            top_k=args.top_k
        )

        token_id = next_token.item()

        generated_ids.append(
            token_id
        )


        # If context is now full, stop.
        if len(generated_ids) >= context_length:
            break


        # IMPORTANT:
        #
        # Only the newly sampled token is passed.
        # Previous K/V tensors come from the cache.
        logits, cache = model.forward_cached(
            next_token,
            cache=cache
        )

        next_logits = logits[:, -1, :]


# =========================================================
# Decode
# =========================================================

text = tokenizer.decode(
    generated_ids
)

print()
print("=" * 70)
print(text)
print("=" * 70)