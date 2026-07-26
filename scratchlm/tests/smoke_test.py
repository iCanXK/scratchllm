from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratchlm.config import ModelConfig
from scratchlm.model import ScratchLM
from scratchlm.tokenizer import ByteTokenizer


def main() -> None:
    tokenizer = ByteTokenizer()
    text = "Hello, 世界!"
    assert tokenizer.decode(tokenizer.encode(text)) == text

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,
    )
    model = ScratchLM(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    output = model(x, x)
    assert output.logits.shape == (2, 16, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()

    generated = model.generate(x[:1, :4], max_new_tokens=4, temperature=1.0, top_k=20)
    assert generated.shape == (1, 8)
    print(f"smoke test passed; parameters={model.num_parameters():,}")


if __name__ == "__main__":
    main()
