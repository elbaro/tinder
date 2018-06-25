import tinder


def test_batch():
    iterable = tinder.batch.iter_from_iter([1, 2, 3, 4], 3)
    iterator = iter(iterable)
    assert next(iterator) == [1, 2, 3]
    assert next(iterator) == [4]
