import redis


class RedisQueue(object):
    def __init__(self, queue: str, redis_client=None):
        self.queue = queue
        if redis_client == None:
            redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
        self.client = redis_client

    def clear(self):
        self.client.delete(self.queue)

    def available(self):
        return self.client.llen(self.queue)

    def push(self, datum):
        self.client.rpush(self.queue, datum)

    def pop_at_least_one(self, max_num: int = 1):
        """
        pop a batch of elements. return at least one.
        Args:
            max_num (int): maximum number of samples to pop.

        Returns:
            a list of elements.
        """

        batch = [self.client.blpop(keys=[self.queue])[1]]

        for _ in range(1, max_num):
            element = self.client.lpop(self.queue)
            if element is None:
                break
            batch.append(element)

        return batch
