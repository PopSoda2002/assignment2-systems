from cs336_systems.benchmark import benchmark_model

def benchmark_script():
    model_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000.0,
    }
    data_config = {
        "batch_size": 1,
        "seq_len": 1024,
    }
    warmup_steps = 0
    num_steps = 10
    benchmark_model(model_config, data_config, warmup_steps, num_steps)

if __name__ == "__main__":
    benchmark_script()