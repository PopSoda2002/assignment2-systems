import triton
import triton.language as tl

import torch
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(NUM_ROWS, D,), strides=(x_stride_row, x_stride_dim), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0,),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, shape=(D,), strides=(weight_stride_dim,), 
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr, shape=(NUM_ROWS,), strides=(output_stride_row,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd(
    x_ptr, weight_ptr,
    grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    grad_output_stride_row,
    grad_x_stride_row, grad_x_stride_dim,
    partial_grad_weight_stride_row, partial_grad_weight_stride_dim,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr, shape=(NUM_ROWS, D,), strides=(x_stride_row, x_stride_dim), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0,),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, shape=(D,), strides=(weight_stride_dim,), 
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr, shape=(NUM_ROWS,), strides=(grad_output_stride_row,), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr, shape=(NUM_ROWS, D,), strides=(grad_x_stride_row, grad_x_stride_dim), 
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0,),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr, shape=(n_row_tiles, D,), strides=(partial_grad_weight_stride_row, partial_grad_weight_stride_dim), 
        offsets=(row_tile_idx, 0,),
        block_shape=(1, D_TILE_SIZE),
        order=(1,0),
    )

    grad_x_buffer = tl.zeros((ROWS_TILE_SIZE, D_TILE_SIZE), dtype=tl.float32)
    partial_grad_weight_buffer = tl.zeros((1, D_TILE_SIZE), dtype=tl.float32)
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_x_buffer += tl.sum(grad_output[:, None] * weight[None, :])
        tl.store(grad_x_block_ptr, grad_x_buffer, boundary_check=(0, 1))

        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        partial_grad_weight_buffer = tl.sum(x * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, partial_grad_weight_buffer, boundary_check=(1,))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

    
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Weight must be 1D and have the same dimension as the input"
        assert x.is_cuda and weight.is_cuda, "Input and weight must be on GPU"
        assert x.is_contiguous(), "Input must be contiguous"
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        n_rows = x.shape[0]
        y = torch.empty((n_rows,), device=x.device)
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1), weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D, ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.reshape(output_dims)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)

        grad_x = torch.empty_like(x)
        weighted_sum_bwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D, ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = torch.sum(partial_grad_weight, axis=0)
        return grad_x, grad_weight

f_weightedsum = WeightedSumFunc.apply
x = torch.randn(5, 10, device="cuda", dtype=torch.float32, requires_grad=True)
weight = torch.randn(10, device="cuda", dtype=torch.float32, requires_grad=True)
y = f_weightedsum(x, weight)
print(y)
y.sum().backward()
print(x.grad)
print(weight.grad)