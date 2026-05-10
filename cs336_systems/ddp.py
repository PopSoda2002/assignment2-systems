import os
import timeit
import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.handles = []
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.post_accumulate_grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad /= self.world_size
        self.handles.clear()

    def post_accumulate_grad_hook(self, p):
        handle = dist.all_reduce(p.grad, async_op=True)
        self.handles.append(handle)