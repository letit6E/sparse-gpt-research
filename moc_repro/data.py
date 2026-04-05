from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from transformers import AutoTokenizer


RAW_TEXT_SOURCES = {
    "tiny_shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "wikitext2_raw": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
}


@dataclass
class TextDataset:
    name: str
    tokenizer_name: str
    text_path: Path
    train_tokens: np.ndarray
    val_tokens: np.ndarray
    vocab_size: int


def _download_text(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target_path.write_text(response.text, encoding="utf-8")
    return target_path


def prepare_contiguous_lm_data(
    source_name: str = "wikitext2_raw",
    tokenizer_name: str = "hf-internal-testing/llama-tokenizer",
    cache_dir: str = "data",
    val_fraction: float = 0.1,
    max_characters: Optional[int] = None,
) -> TextDataset:
    if source_name not in RAW_TEXT_SOURCES:
        raise ValueError(f"Unknown source_name={source_name!r}. Available: {sorted(RAW_TEXT_SOURCES)}")

    text_path = Path(cache_dir) / f"{source_name}.txt"
    _download_text(RAW_TEXT_SOURCES[source_name], text_path)
    text = text_path.read_text(encoding="utf-8")

    if max_characters is not None:
        text = text[:max_characters]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**9
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    token_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    all_tokens = np.asarray(token_ids, dtype=np.int64)

    split_idx = int((1.0 - val_fraction) * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    return TextDataset(
        name=source_name,
        tokenizer_name=tokenizer_name,
        text_path=text_path,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        vocab_size=len(tokenizer),
    )


def sample_batch(
    tokens: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    starts: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    limit = len(tokens) - block_size - 1
    if limit <= 0:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for block_size={block_size}")

    if starts is None:
        starts = torch.randint(0, limit, (batch_size,), generator=generator)

    x = torch.stack(
        [torch.from_numpy(tokens[i : i + block_size]).long() for i in starts.tolist()]
    ).to(device)
    y = torch.stack(
        [torch.from_numpy(tokens[i + 1 : i + block_size + 1]).long() for i in starts.tolist()]
    ).to(device)
    return x, y

