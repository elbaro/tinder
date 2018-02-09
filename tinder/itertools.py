class _batch_known_len(object):
    def __init__(self, iterable, batch_size, total):
        self.iterable = iterable
        self.batch_size = batch_size
        self.total = total
        self.num_batch = ((total-1) // self.batch_size) + 1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        source = iter(self.iterable)

        for i in range(self.num_batch-1):
            chunk = [val for _, val in zip(range(self.batch_size), source)]
            yield chunk

        batch_size = self.total - (self.batch_size*(self.num_batch-1))
        chunk = [val for _, val in zip(range(batch_size), source)]
        yield chunk


class _batch_unknown_len(object):
    def __init__(self, iterable, batch_size):
        self.iterable = iterable
        self.batch_size = batch_size

    def __iter__(self):
        source = iter(self.iterable)

        while True:
            chunk = [val for _, val in zip(range(self.batch_size), source)]
            if not chunk:
                raise StopIteration
            yield chunk


def batch(iterable, batch_size, drop_last=False, total=None):
    """Bachify given iterable.

    if the given iterable has `__len__()`, this returns the iterable with `__len()__`.

    Example::

        import tqdm
        import tinder
        import time


        for i in tqdm.tqdm(tinder.batch(range(100), 4, total=14)):
            print(i)
            time.sleep(0.2)


        def f():
            for i in range(30):
                yield i


        for i in tqdm.tqdm(tinder.batch(f(), 4)):
            print(i)
            time.sleep(0.2)

    Args:
        iterable (iterable)
        batch_size (int)
        total (int, optional): if None, it tries to use `__len()__` of iterable.

    """
    if total == None and hasattr(iterable, '__len__'):
        total = len(iterable)

    if total == None:
        return _batch_unknown_len(iterable, batch_size)
    else:
        return _batch_known_len(iterable, batch_size, total)
