# tinder

[![Build Status](https://travis-ci.org/elbaro/tinder.svg?branch=master)](https://travis-ci.org/elbaro/tinder)

Just extra Pytorch library.
Documentation is [here](http://tinder.readthedocs.org/).


## Usage

`from tinder import *`


## TODO

### sliced_wasserstein_distance (WIP)


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
