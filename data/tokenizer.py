import json
from pathlib import Path
from typing import Dict, List, Tuple


class ByteBPETokenizer:
    """
    Byte-level Byte Pair Encoding tokenizer.

    Token IDs:
        0-255   : raw byte values
        256+    : learned BPE merges
    """

    BASE_VOCAB_SIZE = 256

    def __init__(self):
        # pair -> merged token id
        # Example:
        # (116, 104) -> 256
        # means byte/token sequence "t" + "h" became token 256
        self.merges: Dict[Tuple[int, int], int] = {}

        # token id -> byte sequence represented by that token
        self.vocab: Dict[int, bytes] = {
            i: bytes([i])
            for i in range(self.BASE_VOCAB_SIZE)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -----------------------------------------------------
    # BPE utilities
    # -----------------------------------------------------

    @staticmethod
    def _count_pairs(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count adjacent token pairs.
        """
        counts = {}

        for a, b in zip(ids, ids[1:]):
            pair = (a, b)
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    @staticmethod
    def _merge_pair(
        ids: List[int],
        pair: Tuple[int, int],
        new_id: int
    ) -> List[int]:
        """
        Replace every non-overlapping occurrence of pair
        with new_id, left to right.
        """

        output = []

        i = 0

        while i < len(ids):

            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                output.append(new_id)
                i += 2

            else:
                output.append(ids[i])
                i += 1

        return output

    # -----------------------------------------------------
    # Train tokenizer
    # -----------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
        verbose: bool = True
    ) -> None:

        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least "
                f"{self.BASE_VOCAB_SIZE}"
            )

        # Reset tokenizer
        self.merges = {}

        self.vocab = {
            i: bytes([i])
            for i in range(self.BASE_VOCAB_SIZE)
        }

        # UTF-8 text -> raw bytes -> integer IDs
        ids = list(text.encode("utf-8"))

        num_merges = (
            vocab_size
            - self.BASE_VOCAB_SIZE
        )

        for merge_index in range(num_merges):

            pair_counts = self._count_pairs(ids)

            if not pair_counts:
                break

            # Highest frequency
            best_count = max(pair_counts.values())

            if best_count < min_frequency:
                if verbose:
                    print(
                        f"Stopping early: most frequent pair "
                        f"occurs only {best_count} time(s)."
                    )
                break

            # Deterministic tie breaking:
            # lexicographically smallest pair
            best_pair = min(
                pair
                for pair, count in pair_counts.items()
                if count == best_count
            )

            new_id = (
                self.BASE_VOCAB_SIZE
                + merge_index
            )

            # Record BPE rule
            self.merges[best_pair] = new_id

            # Record what bytes the new token represents
            self.vocab[new_id] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

            # Actually compress training sequence
            ids = self._merge_pair(
                ids,
                best_pair,
                new_id
            )

            if verbose and (
                merge_index % 25 == 0
                or merge_index == num_merges - 1
            ):

                token_bytes = self.vocab[new_id]

                readable = token_bytes.decode(
                    "utf-8",
                    errors="replace"
                )

                print(
                    f"merge {merge_index + 1:3d}/"
                    f"{num_merges} | "
                    f"{best_pair} -> {new_id} | "
                    f"count={best_count:6d} | "
                    f"token={readable!r}"
                )

        if self.vocab_size != vocab_size:
            print(
                f"Warning: requested vocabulary size "
                f"{vocab_size}, but learned "
                f"{self.vocab_size} tokens."
            )

    # -----------------------------------------------------
    # Encode
    # -----------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Text -> UTF-8 bytes -> apply learned BPE merges.
        """

        ids = list(text.encode("utf-8"))

        # Merge rules must be applied in learned order.
        ordered_merges = sorted(
            self.merges.items(),
            key=lambda item: item[1]
        )

        for pair, new_id in ordered_merges:
            ids = self._merge_pair(
                ids,
                pair,
                new_id
            )

        return ids

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    def decode(
        self,
        ids: List[int],
        errors: str = "replace"
    ) -> str:
        """
        Token IDs -> bytes -> UTF-8 text.
        """

        byte_string = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return byte_string.decode(
            "utf-8",
            errors=errors
        )

    # -----------------------------------------------------
    # Persistence
    # -----------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save merge rules.

        The vocabulary itself does not need to be saved:
        it can be reconstructed from the 256 base bytes
        and the merge rules.
        """

        ordered_merges = sorted(
            self.merges.items(),
            key=lambda item: item[1]
        )

        data = {
            "type": "byte_bpe",
            "version": 1,
            "vocab_size": self.vocab_size,
            "merges": [
                [pair[0], pair[1], new_id]
                for pair, new_id in ordered_merges
            ]
        }

        Path(path).write_text(
            json.dumps(
                data,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str):
        """
        Reconstruct tokenizer from saved merge rules.
        """

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        if data["type"] != "byte_bpe":
            raise ValueError(
                "Tokenizer file is not byte-level BPE."
            )

        tokenizer = cls()

        for left, right, new_id in data["merges"]:

            pair = (left, right)

            tokenizer.merges[pair] = new_id

            tokenizer.vocab[new_id] = (
                tokenizer.vocab[left]
                + tokenizer.vocab[right]
            )

        return tokenizer