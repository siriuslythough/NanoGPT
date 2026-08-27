import argparse
import statistics
import time
from pathlib import Path

import torch

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

# If provided, ignore --prompt and take this many
# BPE tokens from the corpus as the benchmark prompt.
parser.add_argument(
    "--prompt-tokens",
    type=int,
    default=None
)

parser.add_argument(
    "--corpus",
    type=str,
    default="input.txt"
)

parser.add_argument(
    "--new-tokens",
    type=int,
    default=64
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=1
)

parser.add_argument(
    "--warmup-runs",
    type=int,
    default=3
)

parser.add_argument(
    "--runs",
    type=int,
    default=10
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
# Load tokenizer
# =========================================================

tokenizer = ByteBPETokenizer.load(
    args.tokenizer
)


# =========================================================
# Load checkpoint + model
# =========================================================

checkpoint = torch.load(
    args.checkpoint,
    map_location=device
)

config = checkpoint["config"]

model = GPT(
    **config
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

context_length = config["context_length"]


if tokenizer.vocab_size != config["vocab_size"]:
    raise ValueError(
        f"Tokenizer vocabulary size "
        f"({tokenizer.vocab_size}) does not match "
        f"model vocabulary size "
        f"({config['vocab_size']})."
    )


# =========================================================
# Construct benchmark prompt
# =========================================================

if args.prompt_tokens is not None:

    if args.prompt_tokens <= 0:
        raise ValueError(
            "--prompt-tokens must be greater than 0."
        )

    corpus_text = Path(
        args.corpus
    ).read_text(
        encoding="utf-8"
    )

    corpus_ids = tokenizer.encode(
        corpus_text
    )

    if len(corpus_ids) < args.prompt_tokens:
        raise ValueError(
            f"Corpus contains only "
            f"{len(corpus_ids)} BPE tokens, "
            f"but {args.prompt_tokens} were requested."
        )

    prompt_ids = corpus_ids[
        :args.prompt_tokens
    ]

else:

    prompt_ids = tokenizer.encode(
        args.prompt
    )


if len(prompt_ids) == 0:
    raise ValueError(
        "Prompt produced zero tokens."
    )


# =========================================================
# Context-length validation
# =========================================================

total_sequence_length = (
    len(prompt_ids)
    + args.new_tokens
)

if total_sequence_length > context_length:

    raise ValueError(
        f"Prompt ({len(prompt_ids)} tokens) + "
        f"generation ({args.new_tokens} tokens) = "
        f"{total_sequence_length}, "
        f"which exceeds model context length "
        f"{context_length}."
    )


# =========================================================
# Make batch
#
# Same prompt repeated across batch.
# This tests true batched tensor inference without needing
# variable-length padding yet.
# =========================================================

single_prompt = torch.tensor(
    prompt_ids,
    dtype=torch.long,
    device=device
).unsqueeze(0)


prompt_batch = single_prompt.repeat(
    args.batch_size,
    1
)


# =========================================================
# CUDA synchronization
#
# CUDA operations are asynchronous.
# Without synchronize(), Python timing would be wrong.
# =========================================================

def synchronize():

    if device == "cuda":
        torch.cuda.synchronize()


# =========================================================
# NAIVE GENERATION
#
# Recomputes the ENTIRE sequence for every new token.
# =========================================================

@torch.inference_mode()
def generate_naive(
    prompt,
    num_new_tokens
):

    tokens = prompt.clone()

    synchronize()

    start = time.perf_counter()

    ttft = None


    for step in range(num_new_tokens):

        # Full sequence gets recomputed every time
        logits = model(tokens)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )


        # Time to first generated token
        if step == 0:

            synchronize()

            ttft = (
                time.perf_counter()
                - start
            )


    synchronize()

    total_time = (
        time.perf_counter()
        - start
    )


    return (
        tokens,
        total_time,
        ttft
    )


# =========================================================
# KV-CACHED GENERATION
#
# Prompt:
#     process once -> populate cache
#
# Each later token:
#     process only newly generated token
# =========================================================

@torch.inference_mode()
def generate_cached(
    prompt,
    num_new_tokens
):

    generated = prompt.clone()

    synchronize()

    start = time.perf_counter()


    # -----------------------------------------------------
    # Prefill
    # -----------------------------------------------------

    logits, cache = model.forward_cached(
        prompt,
        cache=None
    )


    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True
    )


    generated = torch.cat(
        [generated, next_token],
        dim=1
    )


    synchronize()

    ttft = (
        time.perf_counter()
        - start
    )


    # -----------------------------------------------------
    # Incremental decoding
    # -----------------------------------------------------

    for _ in range(
        num_new_tokens - 1
    ):

        # Only ONE new token enters the Transformer.
        # Previous keys/values come from cache.
        logits, cache = model.forward_cached(
            next_token,
            cache=cache
        )


        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )


        generated = torch.cat(
            [generated, next_token],
            dim=1
        )


    synchronize()

    total_time = (
        time.perf_counter()
        - start
    )


    return (
        generated,
        total_time,
        ttft
    )


# =========================================================
# Percentile utility
# =========================================================

def percentile(values, percentile_value):

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        percentile_value / 100
    ) * (len(values) - 1)

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        index - lower
    )

    return (
        values[lower]
        * (1 - fraction)
        +
        values[upper]
        * fraction
    )


# =========================================================
# Warm-up
#
# Important for CUDA:
# - context initialization
# - memory allocations
# - kernel warm-up
# =========================================================

print("\nWarming up GPU...")


for _ in range(
    args.warmup_runs
):

    generate_naive(
        prompt_batch,
        args.new_tokens
    )

    generate_cached(
        prompt_batch,
        args.new_tokens
    )


# =========================================================
# Benchmark
# =========================================================

naive_times = []
cached_times = []

naive_ttfts = []
cached_ttfts = []


for run in range(args.runs):

    # -----------------------------------------------------
    # Naive
    # -----------------------------------------------------

    (
        naive_output,
        naive_time,
        naive_ttft
    ) = generate_naive(
        prompt_batch,
        args.new_tokens
    )


    # -----------------------------------------------------
    # Cached
    # -----------------------------------------------------

    (
        cached_output,
        cached_time,
        cached_ttft
    ) = generate_cached(
        prompt_batch,
        args.new_tokens
    )


    # -----------------------------------------------------
    # Correctness check
    #
    # Greedy decoding means both paths MUST produce
    # exactly identical token sequences.
    # -----------------------------------------------------

    if not torch.equal(
        naive_output,
        cached_output
    ):
        raise AssertionError(
            f"Cached and naive outputs differed "
            f"on benchmark run {run}."
        )


    naive_times.append(
        naive_time
    )

    cached_times.append(
        cached_time
    )

    naive_ttfts.append(
        naive_ttft
    )

    cached_ttfts.append(
        cached_ttft
    )


# =========================================================
# Aggregate latency metrics
# =========================================================

mean_naive = statistics.mean(
    naive_times
)

mean_cached = statistics.mean(
    cached_times
)


mean_naive_ttft = statistics.mean(
    naive_ttfts
)

mean_cached_ttft = statistics.mean(
    cached_ttfts
)


naive_p50 = statistics.median(
    naive_times
)

cached_p50 = statistics.median(
    cached_times
)


naive_p95 = percentile(
    naive_times,
    95
)

cached_p95 = percentile(
    cached_times,
    95
)


# =========================================================
# End-to-end throughput
#
# Includes prefill + all generated tokens.
# =========================================================

total_generated_tokens = (
    args.batch_size
    * args.new_tokens
)


naive_total_throughput = (
    total_generated_tokens
    / mean_naive
)

cached_total_throughput = (
    total_generated_tokens
    / mean_cached
)


total_speedup = (
    cached_total_throughput
    / naive_total_throughput
)


# =========================================================
# Decode-only throughput
#
# Excludes TTFT/prefill.
#
# This is often the more meaningful KV-cache metric because
# cache reuse begins AFTER the first generated token.
# =========================================================

decode_tokens = (
    args.batch_size
    * max(
        args.new_tokens - 1,
        0
    )
)


naive_decode_time = (
    mean_naive
    - mean_naive_ttft
)

cached_decode_time = (
    mean_cached
    - mean_cached_ttft
)


if (
    decode_tokens > 0
    and naive_decode_time > 0
    and cached_decode_time > 0
):

    naive_decode_throughput = (
        decode_tokens
        / naive_decode_time
    )

    cached_decode_throughput = (
        decode_tokens
        / cached_decode_time
    )

    decode_speedup = (
        cached_decode_throughput
        / naive_decode_throughput
    )

else:

    naive_decode_throughput = 0.0
    cached_decode_throughput = 0.0
    decode_speedup = 0.0


# =========================================================
# Print benchmark
# =========================================================

print()
print("=" * 68)
print("INFERENCE BENCHMARK")
print("=" * 68)

print(
    f"Model context:          "
    f"{context_length}"
)

print(
    f"Batch size:             "
    f"{args.batch_size}"
)

print(
    f"Prompt tokens:          "
    f"{len(prompt_ids)}"
)

print(
    f"Generated tokens/seq:   "
    f"{args.new_tokens}"
)

print(
    f"Total generated tokens: "
    f"{total_generated_tokens}"
)

print(
    f"Benchmark runs:         "
    f"{args.runs}"
)


# ---------------------------------------------------------
# Latency
# ---------------------------------------------------------

print()
print("-" * 68)
print("LATENCY")
print("-" * 68)

print(
    f"Naive mean latency:     "
    f"{mean_naive * 1000:.2f} ms"
)

print(
    f"Cached mean latency:    "
    f"{mean_cached * 1000:.2f} ms"
)

print()

print(
    f"Naive p50 latency:      "
    f"{naive_p50 * 1000:.2f} ms"
)

print(
    f"Cached p50 latency:     "
    f"{cached_p50 * 1000:.2f} ms"
)

print()

print(
    f"Naive p95 latency:      "
    f"{naive_p95 * 1000:.2f} ms"
)

print(
    f"Cached p95 latency:     "
    f"{cached_p95 * 1000:.2f} ms"
)


# ---------------------------------------------------------
# TTFT
# ---------------------------------------------------------

print()
print("-" * 68)
print("TIME TO FIRST TOKEN")
print("-" * 68)

print(
    f"Naive TTFT:             "
    f"{mean_naive_ttft * 1000:.2f} ms"
)

print(
    f"Cached TTFT:            "
    f"{mean_cached_ttft * 1000:.2f} ms"
)


# ---------------------------------------------------------
# Total throughput
# ---------------------------------------------------------

print()
print("-" * 68)
print("END-TO-END THROUGHPUT")
print("-" * 68)

print(
    f"Naive throughput:       "
    f"{naive_total_throughput:.2f} tok/s"
)

print(
    f"Cached throughput:      "
    f"{cached_total_throughput:.2f} tok/s"
)

print(
    f"Total KV speedup:       "
    f"{total_speedup:.2f}x"
)


# ---------------------------------------------------------
# Decode throughput
# ---------------------------------------------------------

print()
print("-" * 68)
print("DECODE-ONLY THROUGHPUT")
print("-" * 68)

print(
    f"Naive decode throughput:"
    f" {naive_decode_throughput:.2f} tok/s"
)

print(
    f"Cached decode throughput:"
    f" {cached_decode_throughput:.2f} tok/s"
)

print(
    f"Decode KV speedup:      "
    f"{decode_speedup:.2f}x"
)


# ---------------------------------------------------------
# Correctness
# ---------------------------------------------------------

print()
print("-" * 68)
print("CORRECTNESS")
print("-" * 68)

print(
    "Cached and naive greedy outputs matched exactly "
    "for every benchmark run."
)

print("=" * 68)