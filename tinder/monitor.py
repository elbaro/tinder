class Stat(object):
    # alpha 0 -> smooth, constant
    # alpha 1 -> sharp, window size 1
    def __init__(self, alpha:float):
        self.alpha = alpha
        self.clear()

    def update(self, value, count:int=1):
        self.count += count
        self.sum += value

        value /= count
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha*value + (1-self.alpha)*self.ema

    @property
    def average(self):
        if self.count > 0:
            return self.sum / self.count
        return None

    def clear(self):
        self.ema = None  # ema = Exponential Moving Average
        self.count = 0
        self.sum = 0
