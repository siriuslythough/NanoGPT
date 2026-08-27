import argparse
from pathlib import Path

import torch

from data.tokenizer import ByteBPETokenizer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True
)

parser.add_argument(
    "--vocab-size",
    type=int,
    default=512
)

parser.add_argument(
    "--train-ratio",
    type=float,
    default=0.9
)

parser.add_argument(
    "--tokenizer-output",
    type=str,
    default="tokenizer.json"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="prepared_data"
)

args = parser.parse_args()


# ---------------------------------------------------------
# Read corpus
# ---------------------------------------------------------

text = Path(args.data).read_text(
    encoding="utf-8"
)

if not text:
    raise ValueError(
        f"{args.data} is empty."
    )

print(f"Corpus characters: {len(text):,}")
print(
    f"Corpus UTF-8 bytes: "
    f"{len(text.encode('utf-8')):,}"
)


# ---------------------------------------------------------
# Raw-text train / validation split
# ---------------------------------------------------------

split_index = int(
    len(text) * args.train_ratio
)

train_text = text[:split_index]
val_text = text[split_index:]

print()
print(
    f"Train characters: {len(train_text):,}"
)
print(
    f"Validation characters: {len(val_text):,}"
)


# ---------------------------------------------------------
# Train tokenizer ONLY on training data
# ---------------------------------------------------------

print("\nTraining BPE tokenizer...\n")

tokenizer = ByteBPETokenizer()

tokenizer.train(
    train_text,
    vocab_size=args.vocab_size
)

tokenizer.save(
    args.tokenizer_output
)


# ---------------------------------------------------------
# Encode both splits
# ---------------------------------------------------------

train_ids = tokenizer.encode(train_text)
val_ids = tokenizer.encode(val_text)


# Important invariant
assert tokenizer.decode(
    train_ids,
    errors="strict"
) == train_text

assert tokenizer.decode(
    val_ids,
    errors="strict"
) == val_text


print("\nRound-trip tests passed.")


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

train_tensor = torch.tensor(
    train_ids,
    dtype=torch.long
)

val_tensor = torch.tensor(
    val_ids,
    dtype=torch.long
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_dir = Path(args.output_dir)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    train_tensor,
    output_dir / "train.pt"
)

torch.save(
    val_tensor,
    output_dir / "val.pt"
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

train_bytes = len(
    train_text.encode("utf-8")
)

val_bytes = len(
    val_text.encode("utf-8")
)

print()
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(
    f"Vocabulary size:      "
    f"{tokenizer.vocab_size}"
)

print(
    f"Train tokens:         "
    f"{len(train_ids):,}"
)

print(
    f"Validation tokens:    "
    f"{len(val_ids):,}"
)

print(
    f"Train bytes/token:    "
    f"{train_bytes / len(train_ids):.3f}"
)

print(
    f"Validation bytes/token: "
    f"{val_bytes / len(val_ids):.3f}"
)

print()
print(
    f"Tokenizer -> {args.tokenizer_output}"
)

print(
    f"Train data -> "
    f"{output_dir / 'train.pt'}"
)

print(
    f"Validation data -> "
    f"{output_dir / 'val.pt'}"
)