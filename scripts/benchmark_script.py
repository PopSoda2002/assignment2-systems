from cs336_systems.benchmark import benchmark_model
import cs336_basics.model

def benchmark_script():
    model_config = {
        "vocab_size": 50257,
        "context_length": 16384,
        "d_model": 2048,
        "num_layers": 10,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000.0,
    }
    data_config = {
        "batch_size": 1,
        "seq_len": 1024,
    }
    warmup_steps = 3
    num_steps = 10
    cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention
    profile_memory = True
    benchmark_model(model_config, data_config, warmup_steps, num_steps, profile_memory)

if __name__ == "__main__":
    benchmark_script()