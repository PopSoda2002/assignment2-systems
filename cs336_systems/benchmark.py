import timeit
import torch

from cs336_basics.model import BasicsTransformerLM

def benchmark_model(model_config: dict, data_config: dict, warmup_steps: int = 3, num_steps: int = 10) -> float:
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
    model = BasicsTransformerLM(**model_config)
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

    torch.cuda.synchronize()
    print(f"Starting benchmark for {num_steps} steps...")
    start_time = timeit.default_timer()

    # Forward pass
    with torch.cuda.nvtx.range("forward"):
        for _ in range(num_steps):
            with torch.no_grad():
                y = model(x)
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    avg_time = (end_time - start_time) / num_steps
    print(f"Average time forward pass per step: {avg_time} seconds")

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
