import os
import torch
import torch.distributed as dist

class DDP:
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        for p in self.model.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, input_BLD):
        output_BLD = self.model(input_BLD)
        return output_BLD

    def backward(self, loss):
        loss.backward()
        for p in self.model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        return