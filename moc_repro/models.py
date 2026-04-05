from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .moc import SparseDownProjFunction, SparseTopKMoCFunction, chunked_sparse_down_proj


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out_proj(y)


class DenseSwiGLU(nn.Module):
    def __init__(self, n_embd: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        z = F.silu(g) * u
        return self.dropout(self.down_proj(z))


class MoCSwiGLU(nn.Module):
    def __init__(
        self,
        n_embd: int,
        intermediate_size: int,
        sparsity_ratio: float = 0.75,
        dropout: float = 0.0,
        projection_mode: str = "dense_scatter",
        row_chunk_size: int = 16,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.k_active = max(1, int(intermediate_size * (1.0 - sparsity_ratio)))
        self.projection_mode = projection_mode
        self.row_chunk_size = row_chunk_size
        self.output_chunk_size = 128
        self.active_chunk_size = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        if self.projection_mode == "chunked_sparse":
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            out = chunked_sparse_down_proj(
                z_topk,
                topk_indices,
                self.down_proj.weight,
                output_chunk_size=self.output_chunk_size,
                active_chunk_size=self.active_chunk_size,
            )
        elif self.projection_mode == "dense_scatter":
            z_dense = torch.zeros_like(g)
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            z_dense.scatter_(-1, topk_indices, z_topk)
            out = self.down_proj(z_dense)
        elif self.projection_mode == "sparse_downproj":
            z_topk, topk_indices = SparseTopKMoCFunction.apply(g, u, self.k_active)
            out = SparseDownProjFunction.apply(
                z_topk,
                topk_indices,
                self.down_proj.weight,
                self.row_chunk_size,
            )
        else:
            raise ValueError(f"Unknown projection_mode={self.projection_mode!r}")
        return self.dropout(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        intermediate_size: int,
        dropout: float = 0.0,
        sparse_ffn: bool = False,
        sparsity_ratio: float = 0.75,
        use_checkpoint: bool = False,
        projection_mode: str = "dense_scatter",
        row_chunk_size: int = 16,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.use_checkpoint = use_checkpoint
        self.ffn = (
            MoCSwiGLU(
                n_embd,
                intermediate_size,
                sparsity_ratio=sparsity_ratio,
                dropout=dropout,
                projection_mode=projection_mode,
                row_chunk_size=row_chunk_size,
            )
            if sparse_ffn
            else DenseSwiGLU(n_embd, intermediate_size, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        ffn_input = self.ffn_norm(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.ffn, ffn_input, use_reentrant=False)
        else:
            x = x + self.ffn(ffn_input)
        return x


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    intermediate_size: int
    dropout: float = 0.0
    initializer_range: float = 0.02


class SmallLlamaLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        sparse_ffn: bool = False,
        sparsity_ratio: float = 0.75,
        use_checkpoint: bool = False,
        projection_mode: str = "dense_scatter",
        row_chunk_size: int = 16,
    ):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    sparse_ffn=sparse_ffn,
                    sparsity_ratio=sparsity_ratio,
                    use_checkpoint=use_checkpoint,
                    projection_mode=projection_mode,
                    row_chunk_size=row_chunk_size,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss


def make_dense_and_moc_models(
    config: ModelConfig,
    sparsity_ratio: float = 0.75,
    use_checkpoint: bool = False,
    projection_mode: str = "dense_scatter",
    row_chunk_size: int = 16,
) -> tuple[SmallLlamaLM, SmallLlamaLM]:
    dense_model = SmallLlamaLM(config, sparse_ffn=False, use_checkpoint=use_checkpoint)
    moc_model = SmallLlamaLM(
        config,
        sparse_ffn=True,
        sparsity_ratio=sparsity_ratio,
        use_checkpoint=use_checkpoint,
        projection_mode=projection_mode,
        row_chunk_size=row_chunk_size,
    )
    moc_model.load_state_dict(dense_model.state_dict(), strict=True)
    return dense_model, moc_model


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
