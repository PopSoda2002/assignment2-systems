from __future__ import annotations

import torch
import math
import triton
import triton.language as tl
from einops import einsum

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor(N_q, d_q), K: torch.Tensor(N_k, d_k), V: torch.Tensor(N_k, d_v), is_causal: bool = False):
        B_q, B_k = 16, 16
        Q_batch_size, N_q, d_q = Q.shape
        K_batch_size, N_k, d_k = K.shape
        V_batch_size, N_v, d_v = V.shape
        Q_tiles_count = N_q // B_q
        K_tiles_count = N_k // B_k
        V_tiles_count = K_tiles_count
        Q_tiles = Q.reshape(Q_batch_size, N_q // B_q, B_q, d_q)
        K_tiles = K.reshape(K_batch_size, N_k // B_k, B_k, d_k)
        V_tiles = V.reshape(V_batch_size, N_v // B_k, B_k, d_v)
        O = torch.zeros(Q_batch_size, N_q, d_v)
        L = torch.zeros(Q_batch_size, N_q)
        for i in range(Q_tiles_count):
            Q_i = Q_tiles[:, i, :, :]
            O_i = torch.zeros(Q_batch_size, B_q, d_v)
            l_i = torch.zeros(Q_batch_size, B_q)
            max_i = torch.full((Q_batch_size, B_q), float('-inf'))
            for j in range(K_tiles_count):
                K_j = K_tiles[:, j, :, :]
                V_j = V_tiles[:, j, :, :]
                S_ij = torch.matmul(Q_i, K_j.transpose(1, 2)) / math.sqrt(d_k)
                max_ij = torch.maximum(max_i, torch.max(S_ij, dim=-1).values)
                delta = torch.exp(max_i - max_ij)
                max_i = max_ij
                P_ij = torch.exp(S_ij - max_ij[..., None])
                l_ij = delta * l_i + torch.sum(P_ij, dim=-1)
                l_i = l_ij
                O_ij = delta[..., None] * O_i + torch.matmul(P_ij, V_j)
                O_i = O_ij
            O_i = O_i / l_i[..., None]
            L_i = max_i + torch.log(l_i)
            O[:, i * B_q:(i + 1) * B_q, :] = O_i
            L[:, i * B_q:(i + 1) * B_q] = L_i
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size, Q_len, d_q = Q.shape
        S = einsum(Q, K, "b q d, b k d -> b q k") / math.sqrt(d_q)
        P = torch.exp(S - L[..., None])
        dV = einsum(P, grad_out, "b q k, b q d -> b k d")
        dP = einsum(grad_out, V, "b q d, b k d -> b q k")
        D = (grad_out * O).sum(dim=-1)
        dS = P * (dP - D[..., None])
        dQ = einsum(dS, K, "b q k, b k d -> b q d") / math.sqrt(d_q)
        dK = einsum(dS, Q, "b q k, b q d -> b k d") / math.sqrt(d_q)
        return dQ, dK, dV, None

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr, D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_index = tl.program_id(1)
    if is_causal:
        query_index_range = tl.arange(0, Q_TILE_SIZE) + query_tile_idx * Q_TILE_SIZE
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0,1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    max_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_ij = tl.dot(Q_i, K_j) * scale
        if is_causal:
            key_index_range = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            S_ij = tl.where(key_index_range[None, :] <= query_index_range[:, None], S_ij, -1e7)
        max_ij = tl.maximum(max_i, tl.max(S_ij, axis=-1))
        delta = tl.exp(max_i - max_ij)
        max_i = max_ij
        P_ij = tl.exp(S_ij - max_ij[:, None])
        l_ij = delta * l_i + tl.sum(P_ij, axis=-1)
        l_i = l_ij
        O_ij = delta[:, None] * O_i + tl.dot(P_ij, V_j)
        O_i = O_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    O_i = O_i / l_i[:, None]
    L_i = max_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))
    return

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor(B_q, N_q, d_q), K: torch.Tensor(B_k, N_k, d_k), V: torch.Tensor(B_k, N_k, d_v), is_causal: bool = False):
        batch_size, N_QUERIES, dim = Q.shape
        _, N_KEYS, _ = K.shape
        O = torch.zeros(batch_size, N_QUERIES, dim, device=Q.device)
        L = torch.zeros(batch_size, N_QUERIES, device=Q.device)
        B_q, B_k = 16, 16
        scale = 1 / math.sqrt(dim)
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, B_q), batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale, dim, B_q, B_k, is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    def backward(ctx, grad_out):
        assert NotImplementedError("Backward pass not implemented")