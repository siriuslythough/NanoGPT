import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model.gpt import GPT


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="input.txt")
parser.add_argument("--checkpoint", type=str, default="gpt_char.pt")

parser.add_argument("--context-length", type=int, default=128)
parser.add_argument("--model-dim", type=int, default=128)
parser.add_argument("--num-blocks", type=int, default=4)
parser.add_argument("--num-heads", type=int, default=4)

parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--lr", type=float, default=3e-4)

args = parser.parse_args()


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

torch.manual_seed(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ---------------------------------------------------------
# Load text
# ---------------------------------------------------------

text = Path(args.data).read_text(encoding="utf-8")

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

print(f"Characters: {len(text):,}")
print(f"Vocabulary size: {vocab_size}")


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------

split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]


def get_batch(split_name):

    source = train_data if split_name == "train" else val_data

    indices = torch.randint(
        0,
        len(source) - args.context_length - 1,
        (args.batch_size,)
    )

    x = torch.stack([
        source[i:i + args.context_length]
        for i in indices
    ])

    y = torch.stack([
        source[i + 1:i + args.context_length + 1]
        for i in indices
    ])

    return x.to(device), y.to(device)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

config = {
    "vocab_size": vocab_size,
    "context_length": args.context_length,
    "model_dim": args.model_dim,
    "num_blocks": args.num_blocks,
    "num_heads": args.num_heads,
}

model = GPT(**config).to(device)

num_parameters = sum(p.numel() for p in model.parameters())

print(f"Parameters: {num_parameters / 1e6:.2f}M")


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.01
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def estimate_loss(num_batches=20):

    model.eval()

    losses = {}

    for split_name in ["train", "val"]:

        values = []

        for _ in range(num_batches):

            x, y = get_batch(split_name)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            values.append(loss.item())

        losses[split_name] = sum(values) / len(values)

    model.train()

    return losses


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

model.train()

for step in range(args.steps):

    x, y = get_batch("train")

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Prevent occasional exploding gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    if step % 500 == 0 or step == args.steps - 1:

        losses = estimate_loss()

        print(
            f"step {step:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )


# ---------------------------------------------------------
# Save everything required for inference
# ---------------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": config,
    "stoi": stoi,
    "itos": itos,
}

torch.save(checkpoint, args.checkpoint)

print(f"Saved checkpoint -> {args.checkpoint}")
