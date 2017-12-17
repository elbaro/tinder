# tinder

Just extra Pytorch library.

## Usage
`from tinder import *`

## AssertSize (WIP)
Assert that the input tensor's size is (d1, d2, ..).

```
net = nn.Sequential(
    AssertSize(None, 3, 224, 224),
    nn.Conv2d(3, 64, kernel_size=1, stride=2),
    nn.Conv2d(64, 128, kernel_size=1, stride=2),
    AssertSize(None, 128, 64, 64),
)
```


## Flatten (WIP)
Input: (N, ..)
Output: (N, -1)

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

## ThreadedDataLoader (WIP)
A DataLoader using multithreading instead of multiprocessing.

- Good: No crash with opencv
- Bad: Slow

The curent Pytorch's DataLoader has a crash [issue](https://github.com/opencv/opencv/issues/5150).
It turns out most crashes happen when your loader augments images with opencv and iterates fast.

Opencv has its own threadpool and states. When DataLoader creates multiple workers with fork, it doesn't clone all states.
- pthread's fork only copies the main thread, and opencv thinks it has many threads while in fact it has only one thread.
  some people work around this by temporary solutions. see the above issue.
- OpenCL/CUDA are doing something in background during fork.
 

## LokyDataLoader (WIP)
A DataLoader using a multiprocessing library Loky.

