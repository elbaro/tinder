# tinder

[![Build Status](https://travis-ci.org/elbaro/tinder.svg?branch=master)](https://travis-ci.org/elbaro/tinder)

[![Docs Link](https://img.shields.io/badge/docs-master-orange.svg)](https://elbaro.github.io/tinder)

Pytorch library.


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
