import redis
import time


class RedisQueue(object):
    """
        A FIFO queue based on Redis. Can be used by multiple workers.
        Workers can share the queue by specifying the same queue name.

        Args:
            queue (str): the name of a queue.
            unique_history (bool): Any element that is ever pushed into the queue is not pushed again.
            soft_capacity (int): The max size of queue. This is 'soft' because the queue can grow up to `soft_capacity+(number of workers)-1`.
            redis_client: if not provided, the default one is created.
    """

    def __init__(self, queue: str, unique_history: bool = False, soft_capacity=None, redis_client=None):
        self.queue = queue
        self.unique_history = unique_history
        if self.unique_history:
            self.history_queue = self.queue + '_all'
        if redis_client == None:
            redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
        self.soft_capacity = soft_capacity
        self.cooldown = 0.1
        self.client = redis_client

    def clear(self, history: bool = False):
        """
        Empty the queue.

        Args:
            history: if True, clear all history as well. has no effect if `unique_history` is False.
        """
        self.client.delete(self.queue)
        if history and self.unique_history:
            self.client.delete(self.history_queue)

    def available(self):
        """
        Returns:
            The current size of the queue.
        """
        return self.client.llen(self.queue)

    def push(self, element):
        """
        Append `element` to the queue.

        If `soft_capacity` is set and the queue is full, wait until a room is available.

        If `unique_history` is set, the ever-seen element is ignored.


        Args:
            element: an element of type that redis-py can encode.

        Returns:

        """
        if self.soft_capacity is not None:
            while self.available() > self.soft_capacity:
                if self.cooldown < 2: self.cooldown *= 2
                time.sleep(self.cooldown)
            if self.cooldown > 0.1:
                self.cooldown /= 2

        if self.unique_history:
            if not self.client.sismember(self.history_queue, element):
                self.client.rpush(self.queue, element)
                self.client.sadd(self.history_queue, element)
        else:
            self.client.rpush(self.queue, element)

    def pop_one(self):
        """
        Pop an element.
        Wait if the queue is empty.

        Returns:
            an element.
        """
        return self.client.blpop(keys=[self.queue])[1]

    def pop_at_least_one(self, max_num: int = 1):
        """
        Pop a batch of elements. return at least one.
        Wait if the queue is empty.
        Return immediately if the queue is not empty but smaller than `max_num`.

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
