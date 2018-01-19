def batch(iterable, size):
    source = iter(iterable)
    while True:
        chunk = [val for _, val in zip(range(size), source)]
        if not chunk:
            raise StopIteration
        yield chunk
