from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import torch

from .data import TextDataset, sample_batch
from .models import ModelConfig, SmallLlamaLM, make_dense_and_moc_models


@dataclass
class TrainConfig:
    batch_size: int
    block_size: int
    steps: int
    learning_rate: float = 3e-4
    eval_interval: int = 20
    eval_batches: int = 4
    seed: int = 42


@dataclass
class BenchmarkConfig:
    block_sizes: tuple[int, ...]
    batch_size: int = 8
    steps: int = 10
    learning_rate: float = 3e-4
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _build_starts(
    num_tokens: int,
    batch_size: int,
    block_size: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    limit = num_tokens - block_size - 1
    if limit <= 0:
        raise ValueError(f"Not enough tokens ({num_tokens}) for block_size={block_size}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, limit, (steps, batch_size), generator=generator)


def _estimate_val_loss(
    model: SmallLlamaLM,
    dataset: TextDataset,
    device: torch.device,
    block_size: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> float:
    starts = _build_starts(
        len(dataset.val_tokens),
        batch_size=batch_size,
        block_size=block_size,
        steps=eval_batches,
        seed=seed,
    )
    losses = []
    model.eval()
    with torch.no_grad():
        for step_starts in starts:
            x, y = sample_batch(dataset.val_tokens, block_size, batch_size, device, starts=step_starts)
            _, loss = model(x, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def train_model(
    model: SmallLlamaLM,
    dataset: TextDataset,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    set_seed(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_starts = _build_starts(
        len(dataset.train_tokens),
        batch_size=config.batch_size,
        block_size=config.block_size,
        steps=config.steps,
        seed=config.seed,
    )

    history: list[dict] = []
    _reset_peak_memory(device)
    _device_synchronize(device)
    start_time = time.time()

    for step_idx, step_starts in enumerate(train_starts, start=1):
        x, y = sample_batch(
            dataset.train_tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=device,
            starts=step_starts,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step_idx == 1 or step_idx % config.eval_interval == 0 or step_idx == config.steps:
            val_loss = _estimate_val_loss(
                model,
                dataset,
                device=device,
                block_size=config.block_size,
                batch_size=config.batch_size,
                eval_batches=config.eval_batches,
                seed=config.seed + step_idx,
            )
            history.append(
                {
                    "step": step_idx,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                }
            )

    _device_synchronize(device)
    elapsed = time.time() - start_time
    tokens_per_sec = (config.steps * config.batch_size * config.block_size) / elapsed

    return {
        "history": history,
        "summary": {
            **asdict(config),
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_mb": _peak_memory_mb(device),
        },
    }


def run_pair_training(
    dataset: TextDataset,
    model_config: ModelConfig,
    train_config: TrainConfig,
    sparsity_ratio: float = 0.75,
    use_checkpoint: bool = False,
    device: str | torch.device = "cuda",
) -> dict:
    device = torch.device(device)
    dense_model, moc_model = make_dense_and_moc_models(
        model_config,
        sparsity_ratio=sparsity_ratio,
        use_checkpoint=use_checkpoint,
    )

    dense_run = train_model(dense_model, dataset, train_config, device)
    moc_run = train_model(moc_model, dataset, train_config, device)
    return {"dense": dense_run, "moc": moc_run}


def benchmark_dense_vs_moc(
    dataset: TextDataset,
    model_config: ModelConfig,
    benchmark_config: BenchmarkConfig,
    sparsity_ratio: float = 0.75,
    use_checkpoint: bool = False,
    device: str | torch.device = "cuda",
) -> list[dict]:
    device = torch.device(device)
    rows: list[dict] = []

    for block_size in benchmark_config.block_sizes:
        for model_name, sparse_ffn in (("dense", False), ("moc", True)):
            set_seed(benchmark_config.seed)
            current_config = copy.deepcopy(model_config)
            current_config.block_size = block_size

            if sparse_ffn:
                _, model = make_dense_and_moc_models(
                    current_config,
                    sparsity_ratio=sparsity_ratio,
                    use_checkpoint=use_checkpoint,
                )
            else:
                model, _ = make_dense_and_moc_models(
                    current_config,
                    sparsity_ratio=sparsity_ratio,
                    use_checkpoint=use_checkpoint,
                )

            run_cfg = TrainConfig(
                batch_size=benchmark_config.batch_size,
                block_size=block_size,
                steps=benchmark_config.steps,
                learning_rate=benchmark_config.learning_rate,
                eval_interval=benchmark_config.steps,
                eval_batches=1,
                seed=benchmark_config.seed,
            )

            try:
                result = train_model(model, dataset, run_cfg, device)
                rows.append(
                    {
                        "model": model_name,
                        "block_size": block_size,
                        "status": "ok",
                        "tokens_per_sec": result["summary"]["tokens_per_sec"],
                        "peak_memory_mb": result["summary"]["peak_memory_mb"],
                        "final_train_loss": result["history"][-1]["train_loss"],
                        "final_val_loss": result["history"][-1]["val_loss"],
                    }
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                rows.append(
                    {
                        "model": model_name,
                        "block_size": block_size,
                        "status": "oom",
                        "tokens_per_sec": float("nan"),
                        "peak_memory_mb": float("nan"),
                        "final_train_loss": float("nan"),
                        "final_val_loss": float("nan"),
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return rows


def summarize_histories(pair_runs: dict) -> list[dict]:
    rows: list[dict] = []
    for model_name, payload in pair_runs.items():
        for point in payload["history"]:
            rows.append({"model": model_name, **point})
        rows.append({"model": model_name, "step": "summary", **payload["summary"]})
    return rows
