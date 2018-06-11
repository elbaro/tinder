import torch.utils.data
from collections import Counter

import tinder


def test_balanced():
    ds=[10,20,20,30,30,30,40,40,40,40,50,50,50,50,50]
    loader = tinder.BalancedDataLoader(ds, ds, batch_size=3)
    cnt = Counter()
    for i in range(100):
        for batch in loader:
            cnt.update(batch.numpy())

    # ideally 1:300, 2:300, 3:300, 4:300, 5:300
    for val in cnt.values():
        assert val > 200, val
