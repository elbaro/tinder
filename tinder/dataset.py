import torch.utils.data
from collections import Counter, Iterable


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

    kwargs['shuffle'] = False
    kwargs['batch_sampler'] = None
    kwargs['sampler'] = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

    return torch.utils.data.DataLoader(dataset, **kwargs)
