import timeit
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
from cs336_systems.flashattn import FlashAttentionPyTorch, FlashAttentionTriton

def benchmark_model(model_config: dict, data_config: dict, warmup_steps: int = 3, num_steps: int = 10, profile_memory: bool = False) -> float:
    '''
    transformer model init config:
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10_000.0,
    )
    data_config:
    {"batch_size": int, "seq_len": int}
    '''
    compile_model = True
    model = torch.compile(BasicsTransformerLM(**model_config)) if compile_model else BasicsTransformerLM(**model_config)
    model.eval()
    model.cuda()
    
    batch_size, seq_len = data_config['batch_size'], data_config['seq_len']
    vocab_size = model_config['vocab_size']
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    x = x.cuda()

    # Warmup
    print(f"Warming up for {warmup_steps} steps...")
    for _ in range(warmup_steps):
        with torch.no_grad():
            y = model(x)

    for _ in range(warmup_steps):
        y = model(x)
        y.sum().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    print(f"Starting benchmark for {num_steps} steps...")

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    start_time = timeit.default_timer()

    use_mixed_precision = False
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_mixed_precision else nullcontext()
    # Forward pass
    with torch.cuda.nvtx.range("forward"), ctx, torch.no_grad():
        for _ in range(num_steps):
            y = model(x)
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    avg_time = (end_time - start_time) / num_steps
    print(f"Average time forward pass per step: {avg_time} seconds mixed precision")

    if profile_memory:
        torch.cuda.memory._dump_snapshot("forward_memory_3.pickle")

    # fwd+bwd
    fwd_bwd_start_time = timeit.default_timer()
    with torch.cuda.nvtx.range("forward+backward"):
        for _ in range(num_steps):
            y = model(x)
            y.sum().backward()
            model.zero_grad()
            torch.cuda.synchronize()
    fwd_bwd_end_time = timeit.default_timer()
    fwd_bwd_avg_time = (fwd_bwd_end_time - fwd_bwd_start_time) / num_steps
    print(f"Average time forward+backward pass per step: {fwd_bwd_avg_time} seconds")

    if profile_memory:
        torch.cuda.memory._dump_snapshot("forward+backward_memory_3.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    # full training
    full_train_start_time = timeit.default_timer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with torch.cuda.nvtx.range("full_training"):
        for _ in range(num_steps):
            y = model(x)
            y.sum().backward()
            optimizer.step()
            model.zero_grad()
            torch.cuda.synchronize()
    full_train_end_time = timeit.default_timer()
    full_train_avg_time = (full_train_end_time - full_train_start_time) / num_steps
    print(f"Average time full training per step: {full_train_avg_time} seconds")

def benchmark_attention():
    batch_size = 8
    num_heads = 1
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    num_steps = 10

    compile_sdpa = False
    sdpa_func = torch.compile(scaled_dot_product_attention) if compile_sdpa else scaled_dot_product_attention

    for d_model in d_models:
        for seq_len in seq_lens:
            Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
            K = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
            V = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
            start_time = timeit.default_timer()
            for i in range(num_steps):
                output = sdpa_func(Q, K, V)
            torch.cuda.synchronize()
            mem_before_bwd = torch.cuda.memory_allocated()
            print(f"d_model: {d_model}, seq_len: {seq_len}, Memory before bwd: {mem_before_bwd/1e9:.2f} GB")
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            avg_time = (end_time - start_time) / num_steps
            print(f"d_model: {d_model}, seq_len: {seq_len}, Average time sdpa fwd pass: {avg_time} seconds")

            start_time = timeit.default_timer()
            for i in range(num_steps):
                out = sdpa_func(Q, K, V)
                out.sum().backward()
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            avg_time = (end_time - start_time) / num_steps
            print(f"d_model: {d_model}, seq_len: {seq_len}, Average time sdpa fwd+bwd pass: {avg_time} seconds")

def benchmark_flash_attn(use_triton: bool = False):
    batch_size = 1
    num_steps = 10
    sdpa_func = FlashAttentionTriton.apply if use_triton else FlashAttentionPyTorch.apply
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    d_model = 128
    for seq_len in seq_lens:
        Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
        K = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
        V = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
        start_time = timeit.default_timer()
        for i in range(num_steps):
            output = sdpa_func(Q, K, V)
        torch.cuda.synchronize()
        mem_before_bwd = torch.cuda.memory_allocated()
        print(f"d_model: {d_model}, seq_len: {seq_len}, Memory before bwd: {mem_before_bwd/1e9:.2f} GB")
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        avg_time = (end_time - start_time) / num_steps
        print(f"d_model: {d_model}, seq_len: {seq_len}, Average time sdpa fwd pass: {avg_time} seconds")
        if use_triton:
            continue
        start_time = timeit.default_timer()
        for i in range(num_steps):
            out = sdpa_func(Q, K, V)
            out.sum().backward()
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        avg_time = (end_time - start_time) / num_steps
        print(f"d_model: {d_model}, seq_len: {seq_len}, Average time sdpa fwd+bwd pass: {avg_time} seconds")

def worker_fn(rank, world_size, data_size, num_gpus, warmup_steps):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    length = data_size * 1024 * 1024 // 4
    data = torch.randint(0, 10, (length,), device=f"cuda:{rank}")
    for _ in range(warmup_steps):
        dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    print(f"Average time all reduce {data_size} MB {num_gpus} GPUs per step: {end_time - start_time} seconds")
    cleanup()

def cleanup():
    dist.destroy_process_group()

def benchmark_distributed_communication_single_node():
    # MB
    data_size_ranges = [1, 10, 100, 1024]
    # Number of GPUs
    num_gpus_ranges = [2, 4]
    warmup_steps = 5
    for data_size in data_size_ranges:
        for num_gpus in num_gpus_ranges:
            mp.spawn(fn=worker_fn, args=(num_gpus, data_size, num_gpus, warmup_steps), nprocs=num_gpus, join=True)
