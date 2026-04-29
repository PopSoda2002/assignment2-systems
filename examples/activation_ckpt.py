import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048

block = TransformerBlock(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length),
)

block = torch.compile(block, fullgraph=True)

x = torch.randn((4, context_length, d_model), requires_grad=True)

total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Restoring residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

# # Run forward pass, saving for backward
# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = block(x)
# print(f"Total size of saved tensors in single block pass: {total_size_bytes / (1024**2):.2f} MiB")

# def four_blocks(x):
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     return x

# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = four_blocks(x)
# print(f"Total size of saved tensors in four blocks pass: {total_size_bytes / (1024**2):.2f} MiB")

from torch.utils.checkpoint import checkpoint

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

# Warm up: trigger torch.compile / CUDA initialization outside checkpoint
with torch.no_grad():
    _ = block(x)

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(f"Total size of saved tensors in four blocks checkpoint pass: {total_size_bytes / (1024**2):.2f} MiB")