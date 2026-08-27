import torch

from data.loader import TokenBatchLoader
from data.tokenizer import ByteBPETokenizer


CONTEXT_LENGTH = 128
BATCH_SIZE = 8

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Load prepared data
# ---------------------------------------------------------

train_data = torch.load(
    "prepared_data/train.pt"
)

val_data = torch.load(
    "prepared_data/val.pt"
)


tokenizer = ByteBPETokenizer.load(
    "tokenizer.json"
)


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

loader = TokenBatchLoader(
    train_data=train_data,
    val_data=val_data,
    context_length=CONTEXT_LENGTH,
    batch_size=BATCH_SIZE,
    device=device
)


# ---------------------------------------------------------
# Sample batch
# ---------------------------------------------------------

x, y = loader.get_batch("train")


print("x shape:", x.shape)
print("y shape:", y.shape)

print("x device:", x.device)
print("y device:", y.device)


assert x.shape == (
    BATCH_SIZE,
    CONTEXT_LENGTH
)

assert y.shape == (
    BATCH_SIZE,
    CONTEXT_LENGTH
)


# ---------------------------------------------------------
# Check shift property
# ---------------------------------------------------------

assert torch.equal(
    x[:, 1:],
    y[:, :-1]
)

print("Next-token shift test passed.")


# ---------------------------------------------------------
# Decode one sample
# ---------------------------------------------------------

print("\nInput sample:")
print("-" * 60)

print(
    tokenizer.decode(
        x[0].cpu().tolist()
    )
)

print("-" * 60)

print("\nTarget sample:")
print("-" * 60)

print(
    tokenizer.decode(
        y[0].cpu().tolist()
    )
)

print("-" * 60)