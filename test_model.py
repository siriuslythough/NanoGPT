import torch
import torch.nn.functional as F

from data.tokenizer import ByteBPETokenizer
from model.gpt import GPT


# =========================================================
# Architecture
# =========================================================

CONTEXT_LENGTH = 128
MODEL_DIM = 384
NUM_BLOCKS = 6
NUM_HEADS = 6


# =========================================================
# Load tokenizer + real prepared data
# =========================================================

tokenizer = ByteBPETokenizer.load("tokenizer.json")

VOCAB_SIZE = tokenizer.vocab_size

train_data = torch.load(
    "prepared_data/train.pt"
)


print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Available train tokens: {len(train_data):,}")


# =========================================================
# Build one real next-token sequence
# =========================================================

x = train_data[
    :CONTEXT_LENGTH
].unsqueeze(0)

y = train_data[
    1:CONTEXT_LENGTH + 1
].unsqueeze(0)


print("\nDecoded model input:")
print("-" * 60)

print(
    tokenizer.decode(
        x[0].tolist()
    )
)

print("-" * 60)


# =========================================================
# Model
# =========================================================

model = GPT(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    model_dim=MODEL_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS
)


# =========================================================
# Forward pass
# =========================================================

logits = model(x)


print("\nShapes:")
print("Input:  ", x.shape)
print("Target: ", y.shape)
print("Logits: ", logits.shape)


assert logits.shape == (
    1,
    CONTEXT_LENGTH,
    VOCAB_SIZE
)


# =========================================================
# Loss
# =========================================================

loss = F.cross_entropy(
    logits.reshape(-1, VOCAB_SIZE),
    y.reshape(-1)
)

print(f"\nInitial loss: {loss.item():.4f}")


# =========================================================
# Backprop
# =========================================================

loss.backward()

missing_gradients = []

for name, parameter in model.named_parameters():

    if (
        parameter.requires_grad
        and parameter.grad is None
    ):
        missing_gradients.append(name)


if missing_gradients:
    print(
        "\nParameters without gradients:",
        missing_gradients
    )

else:
    print(
        "\nAll trainable parameters received gradients."
    )


# =========================================================
# Parameter count
# =========================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {num_parameters:,}"
)