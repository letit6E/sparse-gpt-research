# Mixture-of-Channels experiments

PyTorch experiments for dense vs Mixture-of-Channels (MoC) language models: a tiny Llama-style stack, optional transformers Llama integration, and profiling hooks.

## Layout

| Path | Purpose |
|------|--------|
| `moc_repro/` | Data loading, `SmallLlamaLM`, MoC ops, training/benchmarks, HuggingFace Llama wrappers, `torch` profiler utilities |
| `moc_reproducibility_experiment.ipynb` | End-to-end MoC / reproducibility notebook |
| `hf_llama_moc_experiment.ipynb` | HF Llama + MoC patch experiments |
| `first_experiment_sparse_nanogpt.ipynb` | Earlier sparse / NanoGPT-style work |
| `profiling_artifacts/` | Saved profiler traces and summaries (large JSON) |