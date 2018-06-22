import queue

def pop_batch(q, max_batch_size:int):
    assert max_batch_size>0
    batch = [q.get(block=True)]
    for i in range(max_batch_size-1):
        try:
            msg = q.get_nowait()
            batch.append(msg)
        except queue.Empty:
            break

    return batch

