import redis
import pika
import time
import atexit
from typing import List, Iterator, Tuple
import asyncio
from threading import Thread
from multiprocessing import Queue
import queue

from pika.adapters.blocking_connection import BlockingConnection, BlockingChannel
from pika.adapters import AsyncioConnection


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
                if self.cooldown < 2:
                    self.cooldown *= 2
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


class RabbitConsumer(object):
    """
        A RabbitMQ consumer that provides data in batch.

        If the prefetch is 3*B, and you are processing messsages in batch of the size B,
        the server sends you up to 2*B messages in advance.


        Args:
            queue (str): the name of a queue.
            prefetch (int): the number of msgs to prefetch. reommend: batch_size*3
            host (str): the hostname to connect without port.
            port (int): the port to connect

    """

    def __init__(self, queue: str, prefetch: int, host: str = 'localhost', port: int = 5672):
        self.host = host
        self.port = port
        self.queue = queue
        self.prefetch = prefetch
        self.buf = Queue()

        self.thread = Thread(target=self.start_loop, args=(), daemon=True)
        self.thread.start()

        atexit.register(self.close)

    # run on ioloop
    def on_connect(self, unused_connection):
        # TODO: register on connection close callback
        self.conn.channel(on_open_callback=self.on_channel)

    # run on ioloop
    def on_channel(self, channel):
        self.channel = channel
        self.channel.queue_declare(queue=self.queue, callback=self.on_queue_declare)

    # run on ioloop
    def on_queue_declare(self, *args, **kwargs):
        self.channel.basic_qos(prefetch_count=self.prefetch)
        self.channel.basic_consume(self._handle_receive, self.queue)

    # run on new thread
    def start_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.conn = AsyncioConnection(pika.ConnectionParameters(host=self.host, port=self.port, ),
                                      self.on_connect, custom_ioloop=loop)
        self.conn.ioloop.start()  # this triggers connect -> on_connect

    def close(self):
        """
        Close the connection.

        If a msg is not acked until the disconnect, it is considered not delivered.

        """
        # schedule close

        self.channel.close()
        self.thread.join()

    def _handle_receive(self, _channel, method, _header, body):
        body = body.decode()
        self.buf.put((body, method.delivery_tag))

    def one(self, timeout=None):
        """
        Consume one message. (blocking)
        After processing the message, you should call consumer.ack(ack_tag).

        Returns:
            A tuple (msg, ack_tag).

        Raise:
            raise on timeout.
        """
        return self.buf.get(block=True, timeout=timeout)

    def one_batch(self, max_batch_size: int) -> Tuple[List, List]:
        """
        Consume up to `max_batch_size` messages.
        Wait until at least one msg is available.
        After processing the message, you should call consumer.ack.

        Args:
            max_batch_size: the number of messages to consume.

        Returns: (List,List).
        The first is a list of messages.
        The second is a list of ack tags.
        """

        l = [self.buf.get(block=True)]
        for i in range(max_batch_size-1):
            try:
                l.append(self.buf.get(block=False))
            except queue.Empty:
                break

        msgs = [x[0] for x in l]
        acks = [x[1] for x in l]
        return (msgs, acks)

    def iter_batch(self, max_batch_size: int) -> Iterator[Tuple[List, List]]:
        while True:
            yield self.one_batch(max_batch_size)

    def ack(self, ack_tags: List):
        """
        Report that you successfully processed messages.

        Args:
            ack_tags: a single ack tag or a list of ack tags of successful messages.
        """
        #self.conn.loop.call_soon_threadsafe(self._ack, args=(ack_tags,))
        self.conn.loop.call_soon_threadsafe(lambda: self._ack(ack_tags))

    def _ack(self, ack_tags):
        if isinstance(ack_tags, list):
            for tag in ack_tags:
                self.channel.basic_ack(tag)
        else:
            self.channel.basic_ack(ack_tags)

    def ack_upto(self, ack_tag):
        """
        Report that you successfully processed all messages up to `ack_tag`.

        Args:
            ack_tag: the ack tag of the last successful message.
        """
        self.conn.ioloop.call_soon_threadsafe(self._ack_upto, args=(ack_tag,))

    def _ack_upto(self, ack_tag):
        self.channel.basic_ack(ack_tag, multiple=True)

    def nack(self, ack_tags: List, requeue):
        """
        Report that you fail to process messages.

        Args:
            ack_tags: a list of ack tags of successful messages.
        """

        if isinstance(ack_tags, list):
            for tag in ack_tags:
                self.channel.basic_nack(tag)
        else:
            self.channel.basic_nack(ack_tags)

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

        If channel is given, host and port are ignored.
        If channel is not given, host and port are used to create a new channel.

        Args:
            channel (BlockingChannel): the channel instance with ack enabled.
            queue (str): the name of a queue.
    """

    def __init__(self, queue: str, host: str = 'localhost', port: int = 5672,
                 channel: BlockingChannel = None):
        if channel is None:
            self.conn = BlockingConnection(pika.ConnectionParameters(host=host, port=port))
            self.channel = self.conn.channel()
        else:
            self.conn = None
            self.channel = channel

        self.queue = queue
        self.channel.queue_declare(queue=queue)

        # make sure deliveries
        self.channel.confirm_delivery()

    def send(self, msg: str) -> bool:
        """
        send the message. (sync)


        Args:
            msg: a single msg(str)

        Return:
            bool: True on success
        """

        return self.channel.basic_publish(
            '',
            self.queue,
            msg,
            properties=pika.BasicProperties(content_type='text/plain',
                                            delivery_mode=2),  # persistent
            mandatory=True)

    def close(self):
        """
        Close the connection.
        After the call to this method, you cannot `send`.
        """
        if self.conn is not None:
            self.conn.close()
