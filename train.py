import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.loader import TokenBatchLoader
from data.tokenizer import ByteBPETokenizer
from model.gpt import GPT


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()

# Data
parser.add_argument(
    "--train-data",
    type=str,
    default="prepared_data/train.pt"
)

parser.add_argument(
    "--val-data",
    type=str,
    default="prepared_data/val.pt"
)

parser.add_argument(
    "--tokenizer",
    type=str,
    default="tokenizer.json"
)

# Architecture
parser.add_argument(
    "--context-length",
    type=int,
    default=128
)

parser.add_argument(
    "--model-dim",
    type=int,
    default=384
)

parser.add_argument(
    "--num-blocks",
    type=int,
    default=6
)

parser.add_argument(
    "--num-heads",
    type=int,
    default=6
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.2
)

# Training
parser.add_argument(
    "--batch-size",
    type=int,
    default=32
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=5000
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=3e-4
)

parser.add_argument(
    "--min-learning-rate",
    type=float,
    default=3e-5
)

parser.add_argument(
    "--warmup-steps",
    type=int,
    default=200
)

parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.1
)

parser.add_argument(
    "--grad-clip",
    type=float,
    default=1.0
)

# Evaluation
parser.add_argument(
    "--eval-interval",
    type=int,
    default=200
)

parser.add_argument(
    "--eval-batches",
    type=int,
    default=20
)

# Checkpoints
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


# =========================================================
# Device + random seed
# =========================================================

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


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

vocab_size = tokenizer.vocab_size

print(f"Vocabulary size: {vocab_size}")


# =========================================================
# Load prepared data
# =========================================================

train_data = torch.load(
    args.train_data,
    map_location="cpu"
)

val_data = torch.load(
    args.val_data,
    map_location="cpu"
)


print(f"Train tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")


# =========================================================
# Batch loader
# =========================================================

loader = TokenBatchLoader(
    train_data=train_data,
    val_data=val_data,
    context_length=args.context_length,
    batch_size=args.batch_size,
    device=device
)


# =========================================================
# Model
# =========================================================

config = {
    "vocab_size": vocab_size,
    "context_length": args.context_length,
    "model_dim": args.model_dim,
    "num_blocks": args.num_blocks,
    "num_heads": args.num_heads,
    "dropout": args.dropout,
}


model = GPT(
    **config
).to(device)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Parameters: "
    f"{num_parameters:,} "
    f"({num_parameters / 1e6:.2f}M)"
)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)


# =========================================================
# Learning-rate schedule
#
# Linear warmup
#       ↓
# cosine decay
# =========================================================

def get_lr(step):

    # Warmup
    if step < args.warmup_steps:

        return (
            args.learning_rate
            * (step + 1)
            / args.warmup_steps
        )

    # Finished training
    if step >= args.max_steps:

        return args.min_learning_rate

    # Cosine decay
    progress = (
        step - args.warmup_steps
    ) / (
        args.max_steps
        - args.warmup_steps
    )

    coefficient = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        args.min_learning_rate
        + coefficient
        * (
            args.learning_rate
            - args.min_learning_rate
        )
    )


# =========================================================
# Loss function
# =========================================================

def calculate_loss(x, y):

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )

    return loss


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def estimate_losses():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(
            args.eval_batches
        ):

            x, y = loader.get_batch(
                split
            )

            loss = calculate_loss(
                x,
                y
            )

            losses.append(
                loss.item()
            )

        results[split] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# =========================================================
# Checkpoints
# =========================================================

checkpoint_dir = Path(
    args.checkpoint_dir
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)


def save_checkpoint(
    path,
    step,
    best_val_loss
):

    torch.save(
        {
            "step": step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "config":
                config,

            "best_val_loss":
                best_val_loss,

            "tokenizer_path":
                args.tokenizer,
        },
        path
    )


# =========================================================
# Optional mixed precision
# =========================================================

use_amp = (
    device == "cuda"
)

if use_amp:
    scaler = torch.amp.GradScaler(
        "cuda"
    )
else:
    scaler = None


# =========================================================
# Training
# =========================================================

best_val_loss = float("inf")

model.train()

training_start = time.time()


for step in range(
    args.max_steps
):

    step_start = time.time()


    # -----------------------------------------------------
    # Learning rate
    # -----------------------------------------------------

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    # -----------------------------------------------------
    # Batch
    # -----------------------------------------------------

    x, y = loader.get_batch(
        "train"
    )


    # -----------------------------------------------------
    # Forward + backward
    # -----------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    if use_amp:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            loss = calculate_loss(
                x,
                y
            )

        scaler.scale(
            loss
        ).backward()

        # Needed before clipping scaled grads
        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip
        )

        scaler.step(
            optimizer
        )

        scaler.update()


    else:

        loss = calculate_loss(
            x,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip
        )

        optimizer.step()


    # -----------------------------------------------------
    # Basic progress logging
    # -----------------------------------------------------

    if step % 20 == 0:

        step_time = (
            time.time()
            - step_start
        )

        tokens_this_step = (
            args.batch_size
            * args.context_length
        )

        tokens_per_second = (
            tokens_this_step
            / step_time
        )

        print(
            f"step {step:5d} | "
            f"loss {loss.item():.4f} | "
            f"lr {lr:.2e} | "
            f"{tokens_per_second:,.0f} tok/s"
        )


    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    should_evaluate = (
        step % args.eval_interval == 0
        or step == args.max_steps - 1
    )

    if should_evaluate:

        losses = estimate_losses()

        print()
        print(
            f"[evaluation step {step}]"
        )

        print(
            f"train loss: "
            f"{losses['train']:.4f}"
        )

        print(
            f"val loss:   "
            f"{losses['val']:.4f}"
        )


        # Best checkpoint
        if (
            losses["val"]
            < best_val_loss
        ):

            best_val_loss = (
                losses["val"]
            )

            save_checkpoint(
                checkpoint_dir
                / "best.pt",

                step,

                best_val_loss
            )

            print(
                "Saved new best checkpoint."
            )


        # Always maintain latest checkpoint
        save_checkpoint(
            checkpoint_dir
            / "last.pt",

            step,

            best_val_loss
        )

        print()


# =========================================================
# Done
# =========================================================

elapsed = (
    time.time()
    - training_start
)


print(
    f"Training finished in "
    f"{elapsed / 60:.2f} minutes."
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)