import torch.utils.data
from collections import Counter, Iterable
import torch.multiprocessing as tmp
import multiprocessing as mp
from typing import Callable
import queue

import hashlib


def hash100(s: str):
    """
    Hash a string into 1~100.
    Useful when you split a dataset into subsets.
    """
    h = hashlib.md5(s.encode())
    return int(h.hexdigest(), base=16) % 100 + 1


class StreamingDataloader(object):
    """A dataloader for streaming.

    If you have a stream of data (e.g. from RabbitMQ or Kafka), you cannot use a traditional Pytorch Dataset which requires `__len__` to be defined.
    In this case, you can put your streaming data into multiprocessing.Manager().Queue() in the background and pass it to StreamingDataloader.

    - StreamingDataloader is an iterator.
    - `__next__` is blocking and returns at least one element.
    - It never raises `StopIteration`.

    Example::

        import tinder

        def preprocess(msg:str):
            return '(' + msg + ')' + str(len(msg))

        c = tinder.queue.KafkaConsumer(topic='filenames', consumer_id='anonymous_123')

        q = c.start_drain(batch_size=3, capacity=20)
        loader = tinder.dataset.StreamingDataloader(q, batch_size=5, num_workers=2, transform=preprocess)

        for batch in loader:
            print('batch: ', batch)
    """

    def __init__(self, q, batch_size: int, num_workers: int, transform):
        """
        Args:
            q: A thread-safe queue. It should be multiprocessing.Manager().Queue or torch.multiprocessing.Manager().Queue.
            batch_size (int): the maximum size of batch.
            num_workers (int): the number of processes.
            transform: a function that receives a string (msg) and returns any object.
        """

        assert isinstance(q, mp.managers.BaseProxy) or isinstance(
            q, tmp.managers.BaseProxy
        )
        assert batch_size > 0
        assert num_workers > 0

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.m = tmp.Manager()
        self.source = q
        self.sink = self.m.Queue(maxsize=batch_size * 3)

        self.pool = tmp.Pool(num_workers)
        for i in range(num_workers):
            r = self.pool.apply_async(
                self._worker_loop, (self.source, self.sink, transform)
            )

    @staticmethod
    def _worker_loop(source, sink, transform):
        try:
            while True:
                in_ = source.get(block=True)
                out = transform(in_)
                if out is not None:
                    sink.put(out, block=True)
        except Exception as e:
            print("[exception in StreamingDataloader]")
            print(e)

    def __iter__(self):
        return self

    def __next__(self):
        batch = [self.sink.get(block=True)]
        for i in range(self.batch_size - 1):
            try:
                batch.append(self.sink.get_nowait())
            except queue.Empty:
                break
        return batch

    def close(self):
        self.pool.terminate()


def DataLoaderIterator(loader, num=None, last_step=0):
    """Convenient DataLoader wrapper when you need to iterate more than a full batch.

    It is recommended to set `drop_last=True` in your DataLoader.

    Example::

        loader = DataLoader(num_workers=8)
        for step, batch in DataLoaderIterator(loader):
            pass
        for step, _ in DataLoaderIterator(loader, num=6, last_step=2):
            print(step)  # 3, 4, 5, 6


    Args:
        loader (torch.utils.data.DataLoader)
        num (int, optional): Defaults to None. `None` means infinite iteration.
        last_step (int, optional): Defaults to 0. The iteration starts from `last_step+1`.
    """

    step = last_step
    while True:
        for batch in loader:
            step += 1
            if step > num:
                return
            yield step, batch


def BalancedDataLoader(dataset, classes, **kwargs):
    """If your dataset is unbalanced, this wrapper provides a uniform sampling.

    Example::

        # -3 is sampled twice as many as 2 or 3.
        loader = BalancedDataLoader([-3,5,2,3], ['R','G','B','B'], batch_size=1)

    Args:
        dataset (iterable): torch Dataset, list, or any sequence with known length.
        classes (iterable): A list of hashable type. Its length should be equal to the dataset.
        **kwargs: arguments to torch.utils.data.DataLoader
    """

    assert isinstance(classes, Iterable)
    assert len(dataset) == len(classes)
    counter = Counter(classes)
    for key in counter.keys():
        counter[key] = 1.0 / counter[key]
    weights = [counter[cls_] for cls_ in classes]

    kwargs["shuffle"] = False
    kwargs["batch_sampler"] = None
    kwargs["sampler"] = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True
    )

    return torch.utils.data.DataLoader(dataset, **kwargs)


def random_split(dataset, ratio, seed=None):
    """Split a given dataset into several ones, e.g. train/val/test.

    The source of randomness comes from `torch`, which can be fixed by `torch.manual_seed`.

    Arguments:
        dataset {[type]} -- pytorch dataset object

    Keyword Arguments:
        ratio {list} -- A list representing the first n-1 portions.  (example: {[0.7,0.2]} for 70% / 20% / 10%)
    """

    if seed is not None:
        torch.manual_seed(seed)

    n = len(dataset)
    lengths = []
    for r in ratio:
        lengths.append(int(n * r))

    lengths.append(n - sum(lengths))
    return torch.utils.data.random_split(dataset, lengths)
