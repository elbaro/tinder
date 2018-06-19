import redis
import pika
import time


class RedisQueue(object):
    """
        A FIFO queue based on Redis. Can be used by multiple workers.
        Workers can share the queue by specifying the same queue name.

        Example::

            import redis
            import tinder
            q = tinder.serving.RedisQueue('q1')

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


class RabbitBatchConsumer(object):
    """
        A RabbitMQ consumer that provides data in batch.

        Args:
            channel (pika.BlockingChannel): the channel instance with ack enabled.
            queue (str): the name of a queue.
    """

    def __init__(self, queue: str, host: str = None, port: int = None, channel: pika.BlockingChannel = None):
        if channel is None:
            self.conn = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port))
            self.channel = self.conn.channel()
        else:
            self.conn = None
            self.channel = channel

        self.channel.queue_declare(queue=queue)
        self.queue = queue
        self.buf = queue.Queue()

        # start receiving messages
        self.channel.basic_consume(self.handle_delivery, queue=queue)
        self.channel.start_consuming()


    def handle_receive(self, _channel, method, header, body):
        self.buf.put((body, method.delivery_tag))


    def get(self, timeout=None):
        """
        Consume one message. (blocking)
        After processing the message, you should call consumer.ack(ack_tag).

        Returns:
            A tuple (msg, ack_tag).

        Raise:
            raise on timeout.
        """
        return self.buf.get(block=True, timeout=timeout)

        
    def get_batch(self, batch_size) -> (List,List):
        """
        Consume `batch_size` messages. (blocking)
        After processing the message, you should call consumer.ack(ack_tag).

        Args:
            batch_size: the number of messages to consume.

        Returns: (List,List).
        The first is a list of messages.
        The second is a list of ack tags.
        """

        l = [self.buf.get(block=True) for i in range(batch_size)]
        return zip(*l)  # transpose


    def ack(self, ack_tags:List):
        """
        Report that you successfully processed messages.

        Args:
            ack_tags: a list of ack tags of successful messages.
        """

        for tag in ack_tags:
            self.channel.basic_ack(tag)


    def ack_upto(self, ack_tag):
        """
        Report that you successfully processed all messages up to `ack_tag`.

        Args:
            ack_tag: the ack tag of the last successful message.
        """

        self.channel.basic_ack(ack_tag, multiple=True)


    def nack(self, ack_tags:List, requeue):
        """
        Report that you fail to process messages.

        Args:
            ack_tags: a list of ack tags of successful messages.
        """

        for tag in ack_tags:
            self.channel.basic_nack(tag, requeue=requeue)


    def nack_upto(self, ack_tag, requeue):
        """
        Report that you fail to process all messages up to `ack_tag`.

        Args:
            ack_tag: the ack tag of the last successful message.
        """

        self.channel.basic_nack(ack_tag, multiple=True, requeue=requeue)


class RabbitProducer(object):
    """
        A RabbitMQ consumer that provides data in batch.

        Args:
            channel (pika.BlockingChannel): the channel instance with ack enabled.
            queue (str): the name of a queue.
    """

    def __init__(self, queue: str, channel: pika.BlockingChannel):
        self.channel = channel
        self.queue = queue

        self.channel.queue_declare(queue=queue)

        # make sure deliveries
        channel.confirm_delivery()


    def send(self, msg:str) -> bool:
        """
        send the message. (sync)


        Args:
            msg: a single msg(str)

        Return:
            bool: True on success
        """

        return self.channel.basic_publish('',
            self.queue,
            msg,
            properties=pika.BasicProperties(content_type='text/plain',
                                            delivery_mode=2),  # persistent
            mandatory=True)


    def close():
        if self.conn is not None:
            self.conn.close()
