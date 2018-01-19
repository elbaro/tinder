import torch
import torch.nn as nn
from .nn import *
from .dataset import *
from .monitor import *
from .serving import *
from .runner import *
from .itertools import *


class AssertSize(nn.Module):
    """Assert the input has the specified size.

    Example::

        net = nn.Sequential(
            AssertSize(None, 3, 224, 224),
            nn.Conv2d(3, 64, kernel_size=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            AssertSize(None, 128, 64, 64),
        )

    Args:
        size (iterable): an iterable of dimensions. Each dimension is one of -1, None, or positive integer.

    """

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
    """A layer that flattens the input.

    Example::

        net = nn.Sequential(
            nn.Conv2d(..),
            nn.BatchNorm2d(..),
            nn.ReLU(),

            nn.Conv2d(..),
            nn.BatchNorm2d(..),
            nn.ReLU(),

            Flatten(),
            nn.Linear(3*3*512, 1024),
        )

    Args:
        x: input tensor
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class View(nn.Module):
    """nn.Module version of tensor.view().

    Example::

        layer = tinder.View(3, -1, 256)
        x = layer(x)

    The batch dimension is implicit.
    The above code is the same as `tensor.view(tensor.size(0), 3, -1, 256)`.

    Args:
        size_without_batch_dim (iterable): each dimension is one of -1, None, or positive.
    """

    def __init__(self, *size_without_batch_dim):
        super().__init__()
        self.size = [s if s != None else -1 for s in size_without_batch_dim]

    def forward(self, x):
        return x.view(x.size(0), *self.size)


def copy_opt_state(old: torch.optim.Optimizer, new: torch.optim.Optimizer):
    """Copy one optimizer's state to another.

    Example::


    Args:
        old (torch.optim.Optimizer): A source optimizer
        new (torch.optim.Optimizer): A destination optimizer
    """

    for group in new.param_groups:
        for parameter in group['params']:
            if parameter in old.state:
                # e.g. {'step', 'exp_avg', ..}
                new.state[parameter] = old.state[parameter]
