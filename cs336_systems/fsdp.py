import torch
import torch.distributed as dist

class FSDP(torch.nn.Module):

    def __init__(self, module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.handles = []
        self.sharded_params = set()
        self.sharded_params_backup = {}
        for module in self.module.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Embedding):
                full_shape = module.weight.data.shape
                shape_size = full_shape[0] // self.world_size
                start = self.rank * shape_size
                end = (self.rank + 1) * shape_size
                module.weight.data = module.weight.data[start:end].clone()
                self.sharded_params.add(id(module.weight))
                module.register_forward_pre_hook(self.sharded_forward_pre_hook)
                module.register_forward_hook(self.sharded_forward_post_hook)
                module.register_full_backward_pre_hook(self.sharded_backward_pre_hook)
                module.weight.register_post_accumulate_grad_hook(self.sharded_post_accumulate_grad_hook)
            else:
                for p in module.parameters():
                    if p.requires_grad:
                        p.register_post_accumulate_grad_hook(self.replicated_post_accumulate_grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def sharded_forward_pre_hook(self, module, inputs):
        sharded = module.weight.data
        full_shape = (sharded.shape[0] * self.world_size, *sharded.shape[1:])
        full = torch.empty(full_shape, dtype=sharded.dtype, device=sharded.device)
        if self.compute_dtype is not None:
            sharded_temp = sharded.clone().to(self.compute_dtype)
            dist.all_gather_into_tensor(full, sharded_temp)
            del sharded_temp
        else: 
            dist.all_gather_into_tensor(full, sharded)
        self.sharded_params_backup[id(module.weight)] = sharded
        module.weight.data = full

    def sharded_forward_post_hook(self, module, inputs, output):
        sharded = self.sharded_params_backup[id(module.weight)]
        module.weight.data = sharded

    def sharded_backward_pre_hook(self, module, grad_output):
        sharded = module.weight.data
        full_shape = (sharded.shape[0] * self.world_size, *sharded.shape[1:])
        full = torch.empty(full_shape, dtype=sharded.dtype, device=sharded.device)
        dist.all_gather_into_tensor(full, sharded)
        if self.compute_dtype is not None:
            full = full.to(self.compute_dtype)
        self.sharded_params_backup[id(module.weight)] = sharded
        module.weight.data = full

    def sharded_post_accumulate_grad_hook(self, param):
        sharded = self.sharded_params_backup[id(param)]
        shard_grad = torch.empty_like(sharded, dtype=torch.float32)
        grad_fp32 = param.grad.to(torch.float32)
        dist.reduce_scatter_tensor(shard_grad, grad_fp32, op=torch.distributed.ReduceOp.SUM)
        shard_grad.div_(self.world_size)
        param.data = sharded
        param.grad = shard_grad

    def replicated_post_accumulate_grad_hook(self, param):
        handle = dist.all_reduce(param.grad, async_op=True)
        self.handles.append(handle)

    def fsdp_gather_full_params(self):
        out = {}
        for name, p in self.module.named_parameters():
            if id(p) in self.sharded_params:
                sharded = p.data
                full_shape = (sharded.shape[0] * self.world_size, *sharded.shape[1:])
                full = torch.empty(full_shape, dtype=sharded.dtype, device=sharded.device)
                dist.all_gather_into_tensor(full, sharded)
                out[name] = full
            else:
                out[name] = p.data.clone()
        return out

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for p in self.module.parameters():
            if p.requires_grad and id(p) not in self.sharded_params:
                if p.grad is not None:
                    p.grad.div_(self.world_size)
                    if p.grad.dtype != torch.float32:
                        p.grad = p.grad.to(torch.float32)
