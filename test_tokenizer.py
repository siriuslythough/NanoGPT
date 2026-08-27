from data.tokenizer import ByteBPETokenizer


text = """
The quick brown fox jumps over the lazy dog.
Hello, world!
Numbers: 123456789
Symbols: !@#$%^&*()
Unicode: café ₹ नमस्ते 🙂
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
"""


tokenizer = ByteBPETokenizer()

tokenizer.train(
    text,
    vocab_size=300
)


# ---------------------------------------------------------
# Basic encode/decode
# ---------------------------------------------------------

sample = "Hello ₹! café नमस्ते 🙂"

ids = tokenizer.encode(sample)

decoded = tokenizer.decode(
    ids,
    errors="strict"
)

print("Original:")
print(sample)

print()

print("Token IDs:")
print(ids)

print()

print("Decoded:")
print(decoded)

assert decoded == sample

print("\nRound-trip test passed.")


# ---------------------------------------------------------
# Save/load
# ---------------------------------------------------------

tokenizer.save(
    "test_tokenizer.json"
)

loaded = ByteBPETokenizer.load(
    "test_tokenizer.json"
)

ids2 = loaded.encode(sample)
decoded2 = loaded.decode(
    ids2,
    errors="strict"
)

assert ids == ids2
assert decoded2 == sample

print("Save/load test passed.")


# ---------------------------------------------------------
# Compression
# ---------------------------------------------------------

raw_bytes = len(
    sample.encode("utf-8")
)

bpe_tokens = len(ids)

print(
    f"Raw bytes: {raw_bytes}"
)

print(
    f"BPE tokens: {bpe_tokens}"
)

print(
    f"Compression ratio: "
    f"{raw_bytes / bpe_tokens:.2f} bytes/token"
)