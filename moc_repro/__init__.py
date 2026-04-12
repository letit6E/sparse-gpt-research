"""Reusable components for Mixture-of-Channels experiments."""

from .data import TextDataset, prepare_contiguous_lm_data, sample_batch
from .experiment import (
    BenchmarkConfig,
    TrainConfig,
    benchmark_dense_vs_moc,
    run_pair_training,
    summarize_histories,
)
from .hf_llama import HFLlamaConfig, benchmark_hf_llama_pair, patch_hf_llama_mlp_with_moc, train_hf_llama_pair
from .models import ModelConfig, SmallLlamaLM, make_dense_and_moc_models
from .moc import HAS_TRITON, HybridTopKMoCFunction, SparseDownProjFunction, validate_hybrid_topk_moc
from .profiling import profile_hf_pair, profile_hf_train_step
