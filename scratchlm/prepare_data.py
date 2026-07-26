from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scratchlm.tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a UTF-8 corpus into train.bin and val.bin")
    parser.add_argument("input", type=Path, help="Input UTF-8 text file")
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--val-fraction", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must lie strictly between 0 and 1")

    text = args.input.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    tokens = np.asarray(tokenizer.encode(text, add_eos=True), dtype=np.uint16)
    if len(tokens) < 1_000:
        print("Warning: the corpus is very small; generated text will mostly memorize it.")

    split = int(len(tokens) * (1.0 - args.val_fraction))
    split = min(max(split, 1), len(tokens) - 1)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens.tofile(args.out_dir / "train.bin")
    val_tokens.tofile(args.out_dir / "val.bin")

    metadata = {
        "tokenizer": "utf8-byte",
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "source": str(args.input),
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
