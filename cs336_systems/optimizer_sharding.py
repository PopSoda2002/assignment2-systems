from typing import Type
import torch.optim as optim
import torch.distributed as dist

class SharedOptimizer(optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs):
        super().__init__(params, defaults={})
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        all_params = self.param_groups[0]['params']
        params_local = [p for i, p in enumerate(all_params) if i % self.world_size == self.rank]
        self.local_optimizer = optimizer_cls(params_local, **kwargs)

    def step(self, closure=None, **kwargs):
        self.local_optimizer.step(closure, **kwargs)
        all_params = self.param_groups[0]['params']
        for i, p in enumerate(all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group):
        return super().add_param_group(param_group)
