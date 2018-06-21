# import tinder
# import pytest


# def test_redis_queue():
#     q = tinder.RedisQueue('queue_test')
#     q.clear()
#     assert q.available() == 0
#     # with pytest.raises(Exception):
#     #     q.pop_at_least_one()
#     q.push('abc')
#     q.push(3.5)
#     q.push(-3)
#     q.push(4)
#     batch = q.pop_at_least_one(2)
#     assert batch == ['abc', '3.5']
#     batch = q.pop_at_least_one(5)
#     assert batch == ['-3', '4']

#     q.push(0)
#     q.push(0.0)
#     batch = q.pop_at_least_one(3)
#     assert batch == ['0', '0.0']
#     q.clear()

# def test_unique_queue():
#     q = tinder.RedisQueue('unique_queue_test', unique_history=True)
#     q.clear(history=True)
#     assert q.available() == 0
#     # with pytest.raises(Exception):
#     #     q.pop_at_least_one()
#     q.push('abc')
#     q.push(3.5)
#     q.push(3.5)
#     q.push(4)
#     batch = q.pop_at_least_one(4)
#     assert batch == ['abc', '3.5', '4']
#     q.push(0)
#     q.push(0)
#     q.push('abc')
#     batch = q.pop_at_least_one(5)
#     assert batch == ['0']

#     q.push(0)
#     q.push(0.0)
#     batch = q.pop_at_least_one(3)
#     assert batch == ['0.0']
#     q.clear(history=True)
