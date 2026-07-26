from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public-domain Tiny Shakespeare corpus")
    parser.add_argument("--output", type=Path, default=Path("data/tinyshakespeare.txt"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, args.output)
    print(f"Downloaded {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
