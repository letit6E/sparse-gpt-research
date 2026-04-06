from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import TextDataset, sample_batch
from .moc import SparseDownProjFunction, SparseTopKMoCFunction, chunked_sparse_down_proj


@dataclass
class HFLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    attention_dropout: float = 0.0


class HFMoCLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        sparsity_ratio: float = 0.75,
        projection_mode: str = "dense_scatter",
        row_chunk_size: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.k_active = max(1, int(intermediate_size * (1.0 - sparsity_ratio)))
        self.projection_mode = projection_mode
        self.row_chunk_size = row_chunk_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.output_chunk_size = 128
        self.active_chunk_size = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        if self.projection_mode == "chunked_sparse":
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            return chunked_sparse_down_proj(
                z_topk,
                topk_indices,
                self.down_proj.weight,
                output_chunk_size=self.output_chunk_size,
                active_chunk_size=self.active_chunk_size,
            )
        if self.projection_mode == "dense_scatter":
            z_dense = torch.zeros_like(g)
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            z_dense.scatter_(-1, topk_indices, z_topk)
            return self.down_proj(z_dense)
        if self.projection_mode == "sparse_downproj":
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            return SparseDownProjFunction.apply(
                z_topk,
                topk_indices,
                self.down_proj.weight,
                self.row_chunk_size,
            )
        raise ValueError(f"Unknown projection_mode={self.projection_mode!r}")


def patch_hf_llama_mlp_with_moc(
    model: nn.Module,
    sparsity_ratio: float = 0.75,
    projection_mode: str = "dense_scatter",
    row_chunk_size: int = 8,
) -> nn.Module:
    for layer in model.model.layers:
        original_mlp = layer.mlp
        moc_mlp = HFMoCLlamaMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            sparsity_ratio=sparsity_ratio,
            projection_mode=projection_mode,
            row_chunk_size=row_chunk_size,
        )

        with torch.no_grad():
            moc_mlp.gate_proj.weight.copy_(original_mlp.gate_proj.weight)
            moc_mlp.up_proj.weight.copy_(original_mlp.up_proj.weight)
            moc_mlp.down_proj.weight.copy_(original_mlp.down_proj.weight)

        layer.mlp = moc_mlp
    return model


def build_hf_llama_models(
    config: HFLlamaConfig,
    sparsity_ratio: float = 0.75,
    projection_mode: str = "dense_scatter",
    row_chunk_size: int = 8,
):
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        attention_dropout=config.attention_dropout,
    )

    dense_model = LlamaForCausalLM(config=hf_config)
    moc_model = LlamaForCausalLM(config=copy.deepcopy(hf_config))
    moc_model.load_state_dict(dense_model.state_dict(), strict=True)
    moc_model = patch_hf_llama_mlp_with_moc(
        moc_model,
        sparsity_ratio=sparsity_ratio,
        projection_mode=projection_mode,
        row_chunk_size=row_chunk_size,
    )
    return dense_model, moc_model


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
    model: nn.Module,
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
            loss = model(input_ids=x, labels=y).loss
            losses.append(loss.item())
    return sum(losses) / len(losses)


def train_hf_llama_pair(
    dataset: TextDataset,
    model_cfg: HFLlamaConfig,
    block_size: int,
    batch_size: int,
    steps: int,
    learning_rate: float = 3e-4,
    eval_interval: int = 20,
    eval_batches: int = 4,
    seed: int = 42,
    sparsity_ratio: float = 0.75,
    device: str | torch.device = "cuda",
    projection_mode: str = "dense_scatter",
    row_chunk_size: int = 8,
) -> dict:
    device = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dense_model, moc_model = build_hf_llama_models(
        model_cfg,
        sparsity_ratio=sparsity_ratio,
        projection_mode=projection_mode,
        row_chunk_size=row_chunk_size,
    )
    starts = _build_starts(len(dataset.train_tokens), batch_size, block_size, steps, seed)

    def _run_one(model: nn.Module) -> dict:
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        history: list[dict] = []

        _reset_peak_memory(device)
        _device_synchronize(device)
        start_time = time.time()

        for step_idx, step_starts in enumerate(starts, start=1):
            x, y = sample_batch(dataset.train_tokens, block_size, batch_size, device, starts=step_starts)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=x, labels=y).loss
            loss.backward()
            optimizer.step()

            if step_idx == 1 or step_idx % eval_interval == 0 or step_idx == steps:
                val_loss = _estimate_val_loss(
                    model,
                    dataset,
                    device=device,
                    block_size=block_size,
                    batch_size=batch_size,
                    eval_batches=eval_batches,
                    seed=seed + step_idx,
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
        summary = {
            "batch_size": batch_size,
            "block_size": block_size,
            "steps": steps,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "eval_batches": eval_batches,
            "seed": seed,
            "elapsed_sec": elapsed,
            "tokens_per_sec": (steps * batch_size * block_size) / elapsed,
            "peak_memory_mb": _peak_memory_mb(device),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "final_train_ppl": float(torch.exp(torch.tensor(history[-1]["train_loss"]))),
            "final_val_ppl": float(torch.exp(torch.tensor(history[-1]["val_loss"]))),
        }

        model = model.cpu()
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {"history": history, "summary": summary}

    return {
        "dense": _run_one(dense_model),
        "moc": _run_one(moc_model),
    }


def benchmark_hf_llama_pair(
    dataset: TextDataset,
    model_cfg: HFLlamaConfig,
    block_sizes: tuple[int, ...],
    batch_size: int = 4,
    steps: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 42,
    sparsity_ratio: float = 0.75,
    device: str | torch.device = "cuda",
    projection_mode: str = "dense_scatter",
    row_chunk_size: int = 8,
) -> list[dict]:
    device = torch.device(device)
    rows: list[dict] = []

    for block_size in block_sizes:
        for model_name, is_sparse in (("dense", False), ("moc", True)):
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                dense_model, moc_model = build_hf_llama_models(
                    model_cfg,
                    sparsity_ratio=sparsity_ratio,
                    projection_mode=projection_mode,
                    row_chunk_size=row_chunk_size,
                )
                model = moc_model if is_sparse else dense_model
                model = model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                starts = _build_starts(len(dataset.train_tokens), batch_size, block_size, steps, seed)

                _reset_peak_memory(device)
                _device_synchronize(device)
                start_time = time.time()

                for step_starts in starts:
                    x, y = sample_batch(dataset.train_tokens, block_size, batch_size, device, starts=step_starts)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(input_ids=x, labels=y).loss
                    loss.backward()
                    optimizer.step()

                _device_synchronize(device)
                elapsed = time.time() - start_time
                rows.append(
                    {
                        "model": model_name,
                        "block_size": block_size,
                        "status": "ok",
                        "tokens_per_sec": (steps * batch_size * block_size) / elapsed,
                        "peak_memory_mb": _peak_memory_mb(device),
                        "final_train_loss": loss.item(),
                        "final_train_ppl": float(torch.exp(loss.detach().cpu())),
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
                    }
                )
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return rows
