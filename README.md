# tinder

[![Build Status](https://travis-ci.org/elbaro/tinder.svg?branch=master)](https://travis-ci.org/elbaro/tinder)

Just extra Pytorch library.

## Usage

`from tinder import *`

## Basic Layers

### AssertSize

Assert that the input tensor's size is (d1, d2, ..).

```
net = nn.Sequential(
    AssertSize(None, 3, 224, 224),
    nn.Conv2d(3, 64, kernel_size=1, stride=2),
    nn.Conv2d(64, 128, kernel_size=1, stride=2),
    AssertSize(None, 128, 64, 64),
)
```

### Flatten

`tinder.Flatten()`
Flatten to `[N, -1]`.

```
net = nn.Sequential(
    nn.Conv2d(..),
    nn.BatchNorm2d(..),
    nn.ReLU(),

    nn.Conv2d(..),
    nn.BatchNorm2d(..),
    nn.ReLU(),

    Flatten(),
    nn.Linear(3*3*512, 1024),
)
```

### View

`tinder.View(3, -1, 256)`
The batch dimension is implicit.
The above code is the same as `tensor.view(tensor.size(0), 3, -1, 256)`.

## NN

### WeightScale

```
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
ws = WeightScale(conv)
nn.Sequential(
    tinder.WeightScale()
)
```

### PixelwiseNormalize

Input : [N, C, H, W]

```
x = tinder.PixelwiseNormalize()(x)
```

### MinibatchStddev

Input : [N, C, H, W]
Output: [N, C+1, H, W]
`class MinibatchStddev()`
`forward(x)`

## GAN Utility

### wgan_gp

```
gp = tinder.wgan_gp(D, real_img, fake_img)
loss = lambda*gp
```

### sliced_wasserstein_distance (WIP)

.

## Dataset / DataLoader

### DataLoaderIterator

`def DataLoaderIterator(loader, num=None, last_step=0)`
Convenient DataLoader wrapper when you need to iterate more than a full batch.

```py
loader = DataLoader(num_workers=8)
for step, batch in DataLoaderIterator(loader):
    pass
for step, _ in DataLoaderIterator(loader, num=6, last_step=2):
    print(step)  # 3, 4, 5, 6
```

`num=None` means infinite iteration.
It is recommended to set `drop_last=True` in your DataLoader.

### BalancedDataLoader

If your dataset is unbalanced, this wrapper provides a uniform sampling.

`dataset` torch Dataset, list, or any sequence with known length.

`classes` a list of hashable type.

```
def BalancedDataLoader(dataset, classes, **kwargs)
```

```
# -3 is sampled twice as many as 2 or 3.
loader = BalancedDataLoader([-3,5,2,3], ['R','G','B','B'], **kwargs)
```

### ThreadedDataLoader (WIP)

A DataLoader using multithreading instead of multiprocessing.

* Good: No crash with opencv
* Bad: Slow

The curent Pytorch's DataLoader has a crash [issue](https://github.com/opencv/opencv/issues/5150).
It turns out most crashes happen when your loader augments images with opencv and iterates fast.

Opencv has its own threadpool and states. When DataLoader creates multiple workers with fork, it doesn't clone all states.

* pthread's fork only copies the main thread, and opencv thinks it has many threads while in fact it has only one thread.
  some people work around this by temporary solutions. see the above issue.
* OpenCL/CUDA are doing something in background during fork.

### LokyDataLoader (WIP)

A DataLoader using a multiprocessing library Loky.
