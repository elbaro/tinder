import torch

from .nn import *
from .layers import *
from .dataset import *
from .monitor import *
from .serving import *
from .config import *
from .itertools import *


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
