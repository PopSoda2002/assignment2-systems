from __future__ import annotations

import torch
import math


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
        return O
