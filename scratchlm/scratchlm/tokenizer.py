from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ByteTokenizer:
    """A lossless UTF-8 byte tokenizer.

    Token IDs 0..255 represent raw bytes. Two extra tokens are reserved:
    BOS=256 and EOS=257. No tokenizer package or learned vocabulary is needed.
    """

    bos_id: int = 256
    eos_id: int = 257

    @property
    def vocab_size(self) -> int:
        return 258

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids: List[int] = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, errors: str = "replace") -> str:
        data = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id <= 255:
                data.append(token_id)
            elif token_id in (self.bos_id, self.eos_id):
                continue
            else:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary")
        return bytes(data).decode("utf-8", errors=errors)
