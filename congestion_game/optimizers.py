import math
import torch
from torch.optim import SGD

class LangevinSGD(SGD):
    def __init__(self, params, lr=1e-3, tau=1., **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.tau = tau

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            lr = group['lr']
            noise_std = math.sqrt(2.0 * lr * self.tau)
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(noise_std * torch.randn_like(p))
        return loss