import torch


class WarmRestartLR(object):
    """Provide T_cur for the current restart session.

    Example::

        cosine = CosineAnnealingLR(..)
        wr = WarmRestartLR(scheduler=cosine, T_mult=2)

        for epoch in range(100):
            wr.step()
            ..

    Args:
        scheduler: instance of CosineAnnealingLR.
        T_mult (float): ratio to increase T_cur. Default: 2.

    """

    def __init__(self, scheduler, T_mult=2):
        super().__init__()
        self.scheduler = scheduler
        self.T_mult = T_mult

    def step(self):
        if self.scheduler.last_epoch + 1 == self.scheduler.T_max:
            self.scheduler.last_epoch = -1
            self.scheduler.T_max = int(self.scheduler.T_max * self.T_mult)

        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_lr()


def copy_opt_state(old: torch.optim.Optimizer, new: torch.optim.Optimizer):
    """Copy one optimizer's state to another.

    Args:
        old (torch.optim.Optimizer): A source optimizer
        new (torch.optim.Optimizer): A destination optimizer
    """

    for group in new.param_groups:
        for parameter in group["params"]:
            if parameter in old.state:
                # e.g. {'step', 'exp_avg', ..}
                new.state[parameter] = old.state[parameter]
