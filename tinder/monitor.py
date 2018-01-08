from types import SimpleNamespace


class Stat(object):
    """A small class that calculates the statistics for a scalar metric.
    This is useful to calcaulate accuracies and losses across minibatches.

    Args:
        alpha (float): Defaults to 1. Weight(0~1) used for exponential moving average.
            A smaller weight reprsents more smooth average.

    Attributes:
        count (int): the number of samples.
        sum (float)
        average (float): the average of samples. read-only.
        ema (float): exponential moving average.
        alpha (float): coefficient for exponential moving average.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.clear()

    def update(self, value, count: int = 1):
        """Update stats with new samples.

        Args:
            value (int or float): new sample, or average of multiple samples.
            count (int): Defaults to 1. This is useful when batch size is not uniform.
        """

        self.count += count
        self.sum += value

        value /= count
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        return self.ema

    @property
    def average(self) -> float:
        """Average of all samples.

        Returns:
            average
        """

        if self.count > 0:
            return self.sum / self.count
        return None

    def clear(self):
        """Clear the history and reset stats to zero.
        """

        self.ema = None  # ema = Exponential Moving Average
        self.count = 0
        self.sum = 0


class Stats(SimpleNamespace):
    """A class that holds named stats.
    """

    def __init__(self, **stats):
        super().__init__(**stats)

    def clear(self):
        """Clear all stats.
        """

        for stat in self.__dict__.values():
            stat.clear()
