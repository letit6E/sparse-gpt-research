from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
from typing import Iterable

import torch
from torch.autograd.profiler import record_function
from torch.profiler import ProfilerActivity, profile

from .data import TextDataset, prepare_contiguous_lm_data, sample_batch
from .hf_llama import HFLlamaConfig, build_hf_llama_models


def _make_profile_kwargs(activities: list[ProfilerActivity]) -> dict:
    kwargs = {
        "activities": activities,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
    }
    try:
        signature = inspect.signature(profile)
    except (TypeError, ValueError):
        return kwargs
    if "acc_events" in signature.parameters:
        kwargs["acc_events"] = True
    return kwargs


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


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_table(
    prof,
    *,
    sort_by: str,
    row_limit: int,
    only_patterns: Iterable[str] | None = None,
) -> str:
    events = prof.key_averages()
    if only_patterns:
        patterns = tuple(only_patterns)
        events = [evt for evt in events if any(pattern in evt.key for pattern in patterns)]
    if not events:
        return "(no matching events)"
    if only_patterns is None:
        return events.table(sort_by=sort_by, row_limit=row_limit)

    reverse = True
    events = sorted(events, key=lambda evt: getattr(evt, sort_by, 0.0), reverse=reverse)[:row_limit]
    header = f"{'name':60} {'self_cpu_us':>14} {'cpu_total_us':>14} {'self_cuda_us':>14} {'cuda_total_us':>14} {'count':>8}"
    lines = [header, "-" * len(header)]
    for evt in events:
        lines.append(
            f"{evt.key[:60]:60} "
            f"{evt.self_cpu_time_total:14.1f} "
            f"{evt.cpu_time_total:14.1f} "
            f"{getattr(evt, 'self_cuda_time_total', 0.0):14.1f} "
            f"{getattr(evt, 'cuda_time_total', 0.0):14.1f} "
            f"{evt.count:8d}"
        )
    return "\n".join(lines)


def _selected_event_summary(prof, only_patterns: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for evt in prof.key_averages():
        if not any(pattern in evt.key for pattern in only_patterns):
            continue
        rows.append(
            {
                "key": evt.key,
                "cpu_time_total_us": evt.cpu_time_total,
                "self_cpu_time_total_us": evt.self_cpu_time_total,
                "cuda_time_total_us": getattr(evt, "cuda_time_total", 0.0),
                "self_cuda_time_total_us": getattr(evt, "self_cuda_time_total", 0.0),
                "count": evt.count,
            }
        )
    rows.sort(
        key=lambda row: (
            row["self_cuda_time_total_us"],
            row["self_cpu_time_total_us"],
        ),
        reverse=True,
    )
    return rows


def profile_hf_train_step(
    *,
    dataset: TextDataset,
    model_cfg: HFLlamaConfig,
    model_name: str,
    projection_mode: str,
    sparsity_ratio: float,
    row_chunk_size: int,
    block_size: int,
    batch_size: int,
    warmup_steps: int,
    active_steps: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device,
    output_dir: str | Path,
) -> dict:
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dense_model, moc_model = build_hf_llama_models(
        model_cfg,
        sparsity_ratio=sparsity_ratio,
        projection_mode=projection_mode,
        row_chunk_size=row_chunk_size,
    )
    model = dense_model if model_name == "dense" else moc_model
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    starts = _build_starts(
        len(dataset.train_tokens),
        batch_size=batch_size,
        block_size=block_size,
        steps=warmup_steps + active_steps,
        seed=seed,
    )

    for step_starts in starts[:warmup_steps]:
        x, y = sample_batch(dataset.train_tokens, block_size, batch_size, device, starts=step_starts)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=x, labels=y).loss
        loss.backward()
        optimizer.step()
    _device_synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    trace_path = output_dir / f"{model_name}_trace.json"
    summary_path = output_dir / f"{model_name}_summary.txt"
    selected_json_path = output_dir / f"{model_name}_selected_events.json"
    selected_patterns = (
        "moc_",
        "aten::topk",
        "aten::gather",
        "aten::scatter",
        "aten::mm",
        "aten::addmm",
        "aten::bmm",
    )

    with profile(**_make_profile_kwargs(activities)) as prof:
        for step_idx, step_starts in enumerate(starts[warmup_steps:], start=1):
            with record_function(f"profile_{model_name}_train_step"):
                x, y = sample_batch(dataset.train_tokens, block_size, batch_size, device, starts=step_starts)
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=x, labels=y).loss
                loss.backward()
                optimizer.step()
            _device_synchronize(device)
            prof.step()

    prof.export_chrome_trace(str(trace_path))
    overall_table = _make_table(prof, sort_by=sort_by, row_limit=40)
    selected_table = _make_table(
        prof,
        sort_by=sort_by,
        row_limit=80,
        only_patterns=selected_patterns,
    )

    selected_summary = _selected_event_summary(prof, selected_patterns)
    selected_json_path.write_text(json.dumps(selected_summary, indent=2), encoding="utf-8")
    summary_path.write_text(
        "\n".join(
            [
                f"model={model_name}",
                f"projection_mode={projection_mode}",
                f"device={device}",
                "",
                "=== Overall Top Ops ===",
                overall_table,
                "",
                "=== Selected MoC / TopK / GEMM Ops ===",
                selected_table,
                "",
                f"trace_file={trace_path}",
                f"selected_events_json={selected_json_path}",
            ]
        ),
        encoding="utf-8",
    )

    model = model.cpu()
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "selected_events_json": str(selected_json_path),
        "sort_by": sort_by,
    }


def profile_hf_pair(
    *,
    dataset: TextDataset,
    model_cfg: HFLlamaConfig,
    projection_mode: str,
    sparsity_ratio: float,
    row_chunk_size: int,
    block_size: int,
    batch_size: int,
    warmup_steps: int,
    active_steps: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device,
    output_dir: str | Path,
) -> list[dict]:
    rows = []
    for model_name in ("dense", "moc"):
        rows.append(
            profile_hf_train_step(
                dataset=dataset,
                model_cfg=copy.deepcopy(model_cfg),
                model_name=model_name,
                projection_mode=projection_mode,
                sparsity_ratio=sparsity_ratio,
                row_chunk_size=row_chunk_size,
                block_size=block_size,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                active_steps=active_steps,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
                output_dir=output_dir,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile dense vs MoC HF LLaMA train steps.")
    parser.add_argument("--source-name", default="wikitext2_raw")
    parser.add_argument("--tokenizer-name", default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-mode", default="sparse_downproj")
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--sparsity-ratio", type=float, default=0.75)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="profiling_artifacts/hf_llama")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    args = parser.parse_args()

    dataset = prepare_contiguous_lm_data(
        source_name=args.source_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
    )
    model_cfg = HFLlamaConfig(
        vocab_size=dataset.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        attention_dropout=0.0,
    )

    results = profile_hf_pair(
        dataset=dataset,
        model_cfg=model_cfg,
        projection_mode=args.projection_mode,
        sparsity_ratio=args.sparsity_ratio,
        row_chunk_size=args.row_chunk_size,
        block_size=args.block_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
