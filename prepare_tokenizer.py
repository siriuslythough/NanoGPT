import argparse
from pathlib import Path

from data.tokenizer import ByteBPETokenizer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True
)

parser.add_argument(
    "--output",
    type=str,
    default="tokenizer.json"
)

parser.add_argument(
    "--vocab-size",
    type=int,
    default=512
)

args = parser.parse_args()


text = Path(args.data).read_text(
    encoding="utf-8"
)

print(
    f"Corpus characters: {len(text):,}"
)

print(
    f"Corpus UTF-8 bytes: "
    f"{len(text.encode('utf-8')):,}"
)


tokenizer = ByteBPETokenizer()

tokenizer.train(
    text,
    vocab_size=args.vocab_size
)

tokenizer.save(
    args.output
)


encoded = tokenizer.encode(text)

print()
print(
    f"Vocabulary size: {tokenizer.vocab_size}"
)

print(
    f"Encoded tokens: {len(encoded):,}"
)

print(
    f"Average bytes/token: "
    f"{len(text.encode('utf-8')) / len(encoded):.3f}"
)

raw_bytes = len(text.encode("utf-8"))
encoded_tokens = len(encoded)

compression = (
    1 - encoded_tokens / raw_bytes
) * 100

print(
    f"Token reduction vs bytes: "
    f"{compression:.2f}%"
)

print(
    f"Saved tokenizer -> {args.output}"
)