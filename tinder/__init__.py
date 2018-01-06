import torch
import torch.nn as nn
from .nn import *
from .dataset import *
from .monitor import *

class AssertSize(nn.Module):
    # size is a list of dimensions.
    # dimension is a positive number, -1, or None
    def __init__(self, *size):
        super().__init__()
        self.size = [s if s != -1 else None for s in size]

    def __repr__(self):
        return f'AssertSize({self.size})'

    def forward(self, x):
        size = x.size()
        if len(self.size) != len(size):
            raise RuntimeError(f"expected rank {len(self.size)} but got a tensor of rank {len(size)}")

        for expected, given in zip(self.size, size):
            if expected != None and expected != given:
                raise RuntimeError(f"expected size {self.size} but got a tensor of size {size}")


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    def __init__(self, *size_without_batch_dim):
        super().__init__()
        self.size = [s if s != None else -1 for s in size_without_batch_dim]

    def forward(self, x):
        return x.view(x.size(0), *self.size)



def copy_opt_state(old: torch.optim.Optimizer, new: torch.optim.Optimizer):
    for group in new.param_groups:
        for parameter in group['params']:
            if parameter in old.state:
                # e.g. {'step', 'exp_avg', ..}
                new.state[parameter] = old.state[parameter]
