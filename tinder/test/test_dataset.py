import torch.utils.data
from collections import Counter
import multiprocessing as mp
from multiprocessing import Queue

import tinder


def test_balanced():
    ds = [10, 20, 20, 30, 30, 30, 40, 40, 40, 40, 50, 50, 50, 50, 50]
    loader = tinder.dataset.BalancedDataLoader(ds, ds, batch_size=3)
    cnt = Counter()
    for i in range(100):
        for batch in loader:
            cnt.update(batch.numpy())

    # ideally 1:300, 2:300, 3:300, 4:300, 5:300
    for val in cnt.values():
        assert val > 200, val


def transform(x):
    return -x


def test_streaming_dataloader():

    source = mp.Manager().Queue()
    source.put(1)
    source.put(2)
    source.put(3)
    source.put(4)
    source.put(5)

    loader = tinder.dataset.StreamingDataloader(
        source, batch_size=2, num_workers=3, transform=transform
    )
    cnt = 0
    for batch in loader:
        cnt += len(batch)
        if cnt == 5:
            break

    loader.close()
