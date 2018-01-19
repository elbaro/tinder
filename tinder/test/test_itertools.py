import tinder


def test_batch():
    it = tinder.batch([1, 2, 3, 4], 3)
    assert next(it) == [1, 2, 3]
    assert next(it) == [4]
