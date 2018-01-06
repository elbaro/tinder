import torch.utils.data
from collections import Counter, Iterable

def DataLoaderIterator(loader, num=None, last_step=0):
    step = last_step
    while True:
        for batch in loader:
            step += 1
            if step > num:
                return
            yield step, batch


def BalancedDataLoader(dataset, classes, **kwargs):
    assert isinstance(classes, Iterable)
    assert len(dataset) == len(classes)
    counter = Counter(classes)
    for key in counter.keys():
        counter[key] = 1.0/counter[key]
    weights = [counter[cls_] for cls_ in classes]

    kwargs['shuffle'] = False
    kwargs['batch_sampler'] = None
    kwargs['sampler'] = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

    return torch.utils.data.DataLoader(dataset, **kwargs)
