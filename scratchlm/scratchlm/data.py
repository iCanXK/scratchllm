from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class TokenDataset:
    data_dir: Path

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        metadata_path = self.data_dir / "meta.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{metadata_path} does not exist. Run prepare_data.py first."
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: Dict[str, object] = json.load(handle)

        self.train = np.memmap(
            self.data_dir / "train.bin", dtype=np.uint16, mode="r"
        )
        self.val = np.memmap(
            self.data_dir / "val.bin", dtype=np.uint16, mode="r"
        )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if len(data) <= block_size:
            raise ValueError(
                f"{split}.bin has {len(data)} tokens, but block_size is {block_size}. "
                "Use a larger corpus or a smaller block size."
            )

        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64)) for i in starts.tolist()]
        )
        y = torch.stack(
            [torch.from_numpy(np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64)) for i in starts.tolist()]
        )
        return x.to(device), y.to(device)
