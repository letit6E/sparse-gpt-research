from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # cpu for local tests
    triton = None
    tl = None
    HAS_TRITON = False


def _silu_grad(x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return sig + x * sig * (1.0 - sig)


def chunked_sparse_down_proj(
    z_topk: torch.Tensor,
    topk_indices: torch.Tensor,
    down_proj_weight: torch.Tensor,
    output_chunk_size: int = 128,
    active_chunk_size: int = 64,
) -> torch.Tensor:
    batch, seq_len, k_active = z_topk.shape
    hidden_size = down_proj_weight.shape[0]
    intermediate_size = down_proj_weight.shape[1]

    if topk_indices.max().item() >= intermediate_size:
        raise ValueError("topk_indices contain values outside the down_proj input dimension")

    with record_function("moc_chunked_sparse_downproj_forward"):
        out = z_topk.new_zeros(batch, seq_len, hidden_size)
        weight_by_index = down_proj_weight.t().contiguous()

        for h_start in range(0, hidden_size, output_chunk_size):
            h_end = min(h_start + output_chunk_size, hidden_size)
            partial_out = z_topk.new_zeros(batch, seq_len, h_end - h_start)
            weight_chunk = weight_by_index[:, h_start:h_end]

            for k_start in range(0, k_active, active_chunk_size):
                k_end = min(k_start + active_chunk_size, k_active)
                idx_chunk = topk_indices[..., k_start:k_end]
                z_chunk = z_topk[..., k_start:k_end]
                selected_weights = F.embedding(idx_chunk, weight_chunk)
                partial_out = partial_out + torch.einsum("btk,btkh->bth", z_chunk, selected_weights)

            out[..., h_start:h_end] = partial_out

        return out


class SparseDownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_topk: torch.Tensor,
        topk_indices: torch.Tensor,
        down_proj_weight: torch.Tensor,
        row_chunk_size: int = 8,
    ) -> torch.Tensor:
        batch, seq_len, k_active = z_topk.shape
        hidden_size = down_proj_weight.shape[0]
        intermediate_size = down_proj_weight.shape[1]

        if topk_indices.max().item() >= intermediate_size:
            raise ValueError("topk_indices contain values outside the down_proj input dimension")

        use_triton = HAS_TRITON and z_topk.is_cuda and topk_indices.is_cuda and down_proj_weight.is_cuda

        if use_triton:
            with record_function("moc_sparse_downproj_forward_triton"):
                n_rows = batch * seq_len
                z_flat = z_topk.reshape(n_rows, k_active).contiguous()
                idx_flat = topk_indices.reshape(n_rows, k_active).contiguous()
                out_flat = torch.empty(
                    (n_rows, hidden_size),
                    device=z_topk.device,
                    dtype=z_topk.dtype,
                )

                block_h = min(128, triton.next_power_of_2(hidden_size))
                block_k = min(64, triton.next_power_of_2(k_active))
                max_k = triton.next_power_of_2(k_active)

                _sparse_downproj_forward_kernel[(n_rows, triton.cdiv(hidden_size, block_h))](
                    z_flat,
                    idx_flat,
                    down_proj_weight,
                    out_flat,
                    k_active,
                    hidden_size,
                    z_flat.stride(0),
                    z_flat.stride(1),
                    idx_flat.stride(0),
                    idx_flat.stride(1),
                    down_proj_weight.stride(0),
                    down_proj_weight.stride(1),
                    out_flat.stride(0),
                    out_flat.stride(1),
                    BLOCK_H=block_h,
                    BLOCK_K=block_k,
                    MAX_K=max_k,
                )

            ctx.save_for_backward(z_topk, topk_indices, down_proj_weight)
            ctx.row_chunk_size = row_chunk_size
            ctx.use_triton = True
            ctx.block_h = block_h
            ctx.block_k = block_k
            ctx.max_k = max_k
            return out_flat.view(batch, seq_len, hidden_size)

        with record_function("moc_sparse_downproj_forward_fallback"):
            n_rows = batch * seq_len
            z_flat = z_topk.reshape(n_rows, k_active)
            idx_flat = topk_indices.reshape(n_rows, k_active)
            weight_t = down_proj_weight.t().contiguous()

            out_flat = z_topk.new_zeros(n_rows, hidden_size)
            for row_start in range(0, n_rows, row_chunk_size):
                row_end = min(row_start + row_chunk_size, n_rows)
                z_chunk = z_flat[row_start:row_end]
                idx_chunk = idx_flat[row_start:row_end]
                selected_weights = weight_t.index_select(0, idx_chunk.reshape(-1)).view(
                    row_end - row_start, k_active, hidden_size
                )
                out_flat[row_start:row_end] = torch.einsum("rk,rkh->rh", z_chunk, selected_weights)

        ctx.save_for_backward(z_topk, topk_indices, down_proj_weight)
        ctx.row_chunk_size = row_chunk_size
        ctx.use_triton = False
        return out_flat.view(batch, seq_len, hidden_size)

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
        z_topk, topk_indices, down_proj_weight = ctx.saved_tensors

        batch, seq_len, k_active = z_topk.shape
        hidden_size = grad_out.shape[-1]
        n_rows = batch * seq_len

        if getattr(ctx, "use_triton", False):
            z_flat = z_topk.reshape(n_rows, k_active).contiguous()
            idx_flat = topk_indices.reshape(n_rows, k_active).contiguous()
            grad_out_flat = grad_out.reshape(n_rows, hidden_size).contiguous()

            grad_z_flat = torch.empty_like(z_flat)
            grad_weight = torch.zeros_like(down_proj_weight)

            num_h_blocks = triton.cdiv(hidden_size, ctx.block_h)
            num_k_blocks = triton.cdiv(k_active, ctx.block_k)

            with record_function("moc_sparse_downproj_grad_z_triton"):
                _sparse_downproj_grad_z_kernel[(n_rows, num_k_blocks)](
                    grad_out_flat,
                    idx_flat,
                    down_proj_weight,
                    grad_z_flat,
                    k_active,
                    hidden_size,
                    grad_out_flat.stride(0),
                    grad_out_flat.stride(1),
                    idx_flat.stride(0),
                    idx_flat.stride(1),
                    down_proj_weight.stride(0),
                    down_proj_weight.stride(1),
                    grad_z_flat.stride(0),
                    grad_z_flat.stride(1),
                    BLOCK_H=ctx.block_h,
                    BLOCK_K=ctx.block_k,
                    NUM_H_BLOCKS=num_h_blocks,
                )

            with record_function("moc_sparse_downproj_grad_weight_triton"):
                _sparse_downproj_grad_weight_kernel[(n_rows, num_h_blocks)](
                    grad_out_flat,
                    z_flat,
                    idx_flat,
                    grad_weight,
                    k_active,
                    hidden_size,
                    grad_out_flat.stride(0),
                    grad_out_flat.stride(1),
                    z_flat.stride(0),
                    z_flat.stride(1),
                    idx_flat.stride(0),
                    idx_flat.stride(1),
                    grad_weight.stride(0),
                    grad_weight.stride(1),
                    BLOCK_H=ctx.block_h,
                    BLOCK_K=ctx.block_k,
                    MAX_K=ctx.max_k,
                )

            grad_z_topk = grad_z_flat.view(batch, seq_len, k_active)
            return grad_z_topk, None, grad_weight, None

        row_chunk_size = ctx.row_chunk_size
        with record_function("moc_sparse_downproj_backward_fallback"):
            z_flat = z_topk.reshape(n_rows, k_active)
            idx_flat = topk_indices.reshape(n_rows, k_active)
            grad_out_flat = grad_out.reshape(n_rows, hidden_size)
            weight_t = down_proj_weight.t().contiguous()

            grad_z_flat = z_flat.new_zeros(n_rows, k_active)
            grad_weight_t = down_proj_weight.new_zeros(down_proj_weight.shape[1], down_proj_weight.shape[0])

            for row_start in range(0, n_rows, row_chunk_size):
                row_end = min(row_start + row_chunk_size, n_rows)
                z_chunk = z_flat[row_start:row_end]
                idx_chunk = idx_flat[row_start:row_end]
                grad_out_chunk = grad_out_flat[row_start:row_end]

                selected_weights = weight_t.index_select(0, idx_chunk.reshape(-1)).view(
                    row_end - row_start, k_active, hidden_size
                )
                grad_z_flat[row_start:row_end] = torch.einsum("rh,rkh->rk", grad_out_chunk, selected_weights)

                contrib = (z_chunk.unsqueeze(-1) * grad_out_chunk.unsqueeze(1)).reshape(-1, hidden_size)
                grad_weight_t.index_add_(0, idx_chunk.reshape(-1), contrib)

        grad_z_topk = grad_z_flat.view(batch, seq_len, k_active)
        grad_weight = grad_weight_t.t().contiguous()
        return grad_z_topk, None, grad_weight, None


if HAS_TRITON:

    @triton.jit
    def _sparse_downproj_forward_kernel(
        z_ptr,
        idx_ptr,
        weight_ptr,
        out_ptr,
        K,
        H,
        stride_z_row,
        stride_z_k,
        stride_idx_row,
        stride_idx_k,
        stride_w_h,
        stride_w_i,
        stride_out_row,
        stride_out_h,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_K: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        h_block_idx = tl.program_id(1)

        h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for k_start in tl.static_range(0, MAX_K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            idx = tl.load(idx_ptr + row_idx * stride_idx_row + k_offsets * stride_idx_k, mask=k_mask, other=0)
            z_vals = tl.load(z_ptr + row_idx * stride_z_row + k_offsets * stride_z_k, mask=k_mask, other=0.0)

            weight_ptrs = (
                weight_ptr
                + h_offsets[:, None] * stride_w_h
                + idx[None, :] * stride_w_i
            )
            weight_vals = tl.load(weight_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(weight_vals * z_vals[None, :], axis=1)

        tl.store(
            out_ptr + row_idx * stride_out_row + h_offsets * stride_out_h,
            acc,
            mask=h_mask,
        )

    @triton.jit
    def _sparse_downproj_grad_z_kernel(
        grad_out_ptr,
        idx_ptr,
        weight_ptr,
        grad_z_ptr,
        K,
        H,
        stride_go_row,
        stride_go_h,
        stride_idx_row,
        stride_idx_k,
        stride_w_h,
        stride_w_i,
        stride_gz_row,
        stride_gz_k,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_H_BLOCKS: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        k_block_idx = tl.program_id(1)

        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        idx = tl.load(idx_ptr + row_idx * stride_idx_row + k_offsets * stride_idx_k, mask=k_mask, other=0)

        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for h_block in tl.static_range(0, NUM_H_BLOCKS):
            h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H

            grad_out_vals = tl.load(
                grad_out_ptr + row_idx * stride_go_row + h_offsets * stride_go_h,
                mask=h_mask,
                other=0.0,
            )
            weight_ptrs = (
                weight_ptr
                + h_offsets[:, None] * stride_w_h
                + idx[None, :] * stride_w_i
            )
            weight_vals = tl.load(weight_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(weight_vals * grad_out_vals[:, None], axis=0)

        tl.store(
            grad_z_ptr + row_idx * stride_gz_row + k_offsets * stride_gz_k,
            acc,
            mask=k_mask,
        )

    @triton.jit
    def _sparse_downproj_grad_weight_kernel(
        grad_out_ptr,
        z_ptr,
        idx_ptr,
        grad_weight_ptr,
        K,
        H,
        stride_go_row,
        stride_go_h,
        stride_z_row,
        stride_z_k,
        stride_idx_row,
        stride_idx_k,
        stride_gw_h,
        stride_gw_i,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        MAX_K: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        h_block_idx = tl.program_id(1)

        h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H
        grad_out_vals = tl.load(
            grad_out_ptr + row_idx * stride_go_row + h_offsets * stride_go_h,
            mask=h_mask,
            other=0.0,
        )

        for k_start in tl.static_range(0, MAX_K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K

            idx = tl.load(idx_ptr + row_idx * stride_idx_row + k_offsets * stride_idx_k, mask=k_mask, other=0)
            z_vals = tl.load(z_ptr + row_idx * stride_z_row + k_offsets * stride_z_k, mask=k_mask, other=0.0)
            contrib = grad_out_vals[:, None] * z_vals[None, :]

            grad_weight_ptrs = (
                grad_weight_ptr
                + h_offsets[:, None] * stride_gw_h
                + idx[None, :] * stride_gw_i
            )
            tl.atomic_add(grad_weight_ptrs, contrib, mask=h_mask[:, None] & k_mask[None, :])

    @triton.jit
    def _topk_backward_kernel(
        grad_z_ptr,
        topk_indices_ptr,
        topk_vals_ptr,
        u_topk_ptr,
        grad_g_ptr,
        grad_u_ptr,
        K,
        stride_row_z,
        stride_row_k,
        BLOCK_K: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        grad_z_row = grad_z_ptr + row_idx * stride_row_z
        grad_g_row = grad_g_ptr + row_idx * stride_row_z
        grad_u_row = grad_u_ptr + row_idx * stride_row_z

        idx_row = topk_indices_ptr + row_idx * stride_row_k
        g_row = topk_vals_ptr + row_idx * stride_row_k
        u_row = u_topk_ptr + row_idx * stride_row_k

        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K

        indices = tl.load(idx_row + offsets, mask=mask)
        g_vals = tl.load(g_row + offsets, mask=mask)
        u_vals = tl.load(u_row + offsets, mask=mask)
        grad_z_vals = tl.load(grad_z_row + indices, mask=mask)

        sig = 1.0 / (1.0 + tl.exp(-g_vals))
        silu_val = g_vals * sig
        silu_grad = sig + silu_val * (1.0 - sig)

        grad_u_vals = grad_z_vals * silu_val
        grad_g_vals = grad_z_vals * u_vals * silu_grad

        tl.store(grad_g_row + indices, grad_g_vals, mask=mask)
        tl.store(grad_u_row + indices, grad_u_vals, mask=mask)

    @triton.jit
    def _topk_sparse_backward_kernel(
        grad_z_topk_ptr,
        topk_indices_ptr,
        topk_vals_ptr,
        u_topk_ptr,
        grad_g_ptr,
        grad_u_ptr,
        K,
        stride_row_z,
        stride_row_k,
        BLOCK_K: tl.constexpr,
    ):
        row_idx = tl.program_id(0)

        grad_z_row = grad_z_topk_ptr + row_idx * stride_row_z
        grad_g_row = grad_g_ptr + row_idx * stride_row_k
        grad_u_row = grad_u_ptr + row_idx * stride_row_k

        idx_row = topk_indices_ptr + row_idx * stride_row_k
        g_row = topk_vals_ptr + row_idx * stride_row_k
        u_row = u_topk_ptr + row_idx * stride_row_k

        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K

        indices = tl.load(idx_row + offsets, mask=mask)
        g_vals = tl.load(g_row + offsets, mask=mask)
        u_vals = tl.load(u_row + offsets, mask=mask)
        grad_z_vals = tl.load(grad_z_row + offsets, mask=mask)

        sig = 1.0 / (1.0 + tl.exp(-g_vals))
        silu_val = g_vals * sig
        silu_grad = sig + silu_val * (1.0 - sig)

        grad_u_vals = grad_z_vals * silu_val
        grad_g_vals = grad_z_vals * u_vals * silu_grad

        tl.store(grad_g_row + indices, grad_g_vals, mask=mask)
        tl.store(grad_u_row + indices, grad_u_vals, mask=mask)


class ReferenceTopKMoCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, k_active: int) -> torch.Tensor:
        with record_function("moc_topk_forward_reference"):
            topk_vals, topk_indices = torch.topk(g, k_active, dim=-1)
            u_topk = torch.gather(u, -1, topk_indices)
            z_topk = F.silu(topk_vals) * u_topk

        ctx.save_for_backward(topk_indices, topk_vals, u_topk)
        z_out = torch.zeros_like(g)
        z_out.scatter_(-1, topk_indices, z_topk)
        return z_out

    @staticmethod
    def backward(ctx, grad_z_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        topk_indices, topk_vals, u_topk = ctx.saved_tensors
        grad_z_topk = torch.gather(grad_z_out, -1, topk_indices)

        grad_u_topk = grad_z_topk * F.silu(topk_vals)
        grad_g_topk = grad_z_topk * u_topk * _silu_grad(topk_vals)

        grad_g = torch.zeros_like(grad_z_out)
        grad_u = torch.zeros_like(grad_z_out)
        grad_g.scatter_(-1, topk_indices, grad_g_topk)
        grad_u.scatter_(-1, topk_indices, grad_u_topk)
        return grad_g, grad_u, None


class HybridTopKMoCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, k_active: int) -> torch.Tensor:
        with record_function("moc_topk_forward_hybrid"):
            topk_vals, topk_indices = torch.topk(g, k_active, dim=-1)
            u_topk = torch.gather(u, -1, topk_indices)
            z_topk = F.silu(topk_vals) * u_topk

        ctx.save_for_backward(topk_indices, topk_vals, u_topk)
        ctx.k_active = k_active
        ctx.use_triton = HAS_TRITON and g.is_cuda and u.is_cuda

        z_out = torch.zeros_like(g)
        z_out.scatter_(-1, topk_indices, z_topk)
        return z_out

    @staticmethod
    def backward(ctx, grad_z_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        topk_indices, topk_vals, u_topk = ctx.saved_tensors

        if not ctx.use_triton:
            with record_function("moc_topk_backward_hybrid_fallback"):
                grad_z_topk = torch.gather(grad_z_out, -1, topk_indices)
                grad_u_topk = grad_z_topk * F.silu(topk_vals)
                grad_g_topk = grad_z_topk * u_topk * _silu_grad(topk_vals)

                grad_g = torch.zeros_like(grad_z_out)
                grad_u = torch.zeros_like(grad_z_out)
                grad_g.scatter_(-1, topk_indices, grad_g_topk)
                grad_u.scatter_(-1, topk_indices, grad_u_topk)
                return grad_g, grad_u, None

        k_active = ctx.k_active
        batch, seq_len, hidden = grad_z_out.shape
        n_rows = batch * seq_len

        grad_g = torch.zeros_like(grad_z_out)
        grad_u = torch.zeros_like(grad_z_out)

        block_k = triton.next_power_of_2(k_active)
        with record_function("moc_topk_backward_hybrid_triton"):
            _topk_backward_kernel[(n_rows,)](
                grad_z_out.view(n_rows, hidden),
                topk_indices.view(n_rows, k_active),
                topk_vals.view(n_rows, k_active),
                u_topk.view(n_rows, k_active),
                grad_g.view(n_rows, hidden),
                grad_u.view(n_rows, hidden),
                k_active,
                grad_z_out.view(n_rows, hidden).stride(0),
                topk_indices.view(n_rows, k_active).stride(0),
                BLOCK_K=block_k,
            )
        return grad_g, grad_u, None


class SparseTopKMoCFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        g: torch.Tensor,
        u: torch.Tensor,
        k_active: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with record_function("moc_topk_forward_sparse"):
            topk_vals, topk_indices = torch.topk(g, k_active, dim=-1)
            u_topk = torch.gather(u, -1, topk_indices)
            z_topk = F.silu(topk_vals) * u_topk

        ctx.save_for_backward(topk_indices, topk_vals, u_topk)
        ctx.k_active = k_active
        ctx.hidden = g.shape[-1]
        ctx.use_triton = HAS_TRITON and g.is_cuda and u.is_cuda
        return z_topk, topk_indices

    @staticmethod
    def backward(
        ctx,
        grad_z_topk: torch.Tensor,
        grad_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del grad_indices
        topk_indices, topk_vals, u_topk = ctx.saved_tensors

        if not ctx.use_triton:
            with record_function("moc_topk_backward_sparse_fallback"):
                grad_u_topk = grad_z_topk * F.silu(topk_vals)
                grad_g_topk = grad_z_topk * u_topk * _silu_grad(topk_vals)

                grad_g = torch.zeros(
                    (*grad_z_topk.shape[:-1], ctx.hidden),
                    device=grad_z_topk.device,
                    dtype=grad_z_topk.dtype,
                )
                grad_u = torch.zeros_like(grad_g)
                grad_g.scatter_(-1, topk_indices, grad_g_topk)
                grad_u.scatter_(-1, topk_indices, grad_u_topk)
                return grad_g, grad_u, None

        k_active = ctx.k_active
        batch, seq_len, _ = grad_z_topk.shape
        n_rows = batch * seq_len

        grad_g = torch.zeros(
            (batch, seq_len, ctx.hidden),
            device=grad_z_topk.device,
            dtype=grad_z_topk.dtype,
        )
        grad_u = torch.zeros_like(grad_g)

        block_k = triton.next_power_of_2(k_active)
        with record_function("moc_topk_backward_sparse_triton"):
            _topk_sparse_backward_kernel[(n_rows,)](
                grad_z_topk.view(n_rows, k_active),
                topk_indices.view(n_rows, k_active),
                topk_vals.view(n_rows, k_active),
                u_topk.view(n_rows, k_active),
                grad_g.view(n_rows, ctx.hidden),
                grad_u.view(n_rows, ctx.hidden),
                k_active,
                grad_z_topk.view(n_rows, k_active).stride(0),
                topk_indices.view(n_rows, k_active).stride(0),
                BLOCK_K=block_k,
            )
        return grad_g, grad_u, None


@dataclass
class ValidationResult:
    max_forward_abs_diff: float
    max_grad_g_abs_diff: float
    max_grad_u_abs_diff: float


def validate_hybrid_topk_moc(
    device: str = "cuda",
    shape: tuple[int, int, int] = (2, 4, 32),
    k_active: int = 8,
    seed: int = 0,
) -> ValidationResult:
    torch.manual_seed(seed)
    g_ref = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    u_ref = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
    g_hybrid = g_ref.detach().clone().requires_grad_(True)
    u_hybrid = u_ref.detach().clone().requires_grad_(True)

    out_ref = ReferenceTopKMoCFunction.apply(g_ref, u_ref, k_active)
    out_hybrid = HybridTopKMoCFunction.apply(g_hybrid, u_hybrid, k_active)

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_hybrid.backward(grad_out)

    return ValidationResult(
        max_forward_abs_diff=(out_ref - out_hybrid).abs().max().item(),
        max_grad_g_abs_diff=(g_ref.grad - g_hybrid.grad).abs().max().item(),
        max_grad_u_abs_diff=(u_ref.grad - u_hybrid.grad).abs().max().item(),
    )
