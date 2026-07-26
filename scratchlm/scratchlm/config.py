from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    vocab_size: int = 258
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10_000.0
    use_sdpa: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_dim = self.n_embd // self.n_head
        if head_dim % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModelConfig":
        return cls(**values)


PRESETS: Dict[str, Dict[str, Any]] = {
    # Approximately 3.4M parameters.
    "tiny": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
    # Approximately 10.8M parameters.
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    # Approximately 35M parameters; substantially slower on a laptop.
    "medium": dict(n_layer=8, n_head=8, n_embd=640, block_size=512),
}


def config_from_preset(name: str, **overrides: Any) -> ModelConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    values = dict(PRESETS[name])
    values.update(overrides)
    return ModelConfig(**values)
