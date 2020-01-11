try:
    import redis
except ImportError:
    _has_redis = False
else:
    _has_redis = True

try:
    from pika.adapters.blocking_connection import BlockingConnection, BlockingChannel
    from pika.adapters import AsyncioConnection
except ImportError:
    _has_pika = False
else:
    _has_pika = True

try:
    import asyncio
    from nats.aio.client import Client as NATS
    from stan.aio.client import Client as STAN
    import stan
    import stan.pb.protocol_pb2 as protocol
except ImportError:
    _has_nats = False
else:
    _has_nats = True

try:
    from confluent_kafka import Producer, Consumer
except ImportError:
    _has_kafka = False
else:
    _has_kafka = True

import time
import atexit
from typing import List, Iterator, Tuple
from threading import Thread
import multiprocessing as mp
import multiprocessing.managers
from multiprocessing import Queue, Process
import queue
from .batch import pop
from .utils import WaitGroup


if _has_redis:

    class RedisQueue(object):
        """
            A FIFO queue based on Redis. Can be used by multiple workers.
            Workers can share the queue by specifying the same queue name.

            Example::

                import redis
                import tinder
                q = tinder.queue.RedisQueue('q1')

            Args:
                queue (str): the name of a queue.
                unique_history (bool): Any element that is ever pushed into the queue is not pushed again.
                soft_capacity (int): The max size of queue. This is 'soft' because the queue can grow up to `soft_capacity+(number of workers)-1`.
                redis_client: if not provided, the default one is created.
        """

        def __init__(
            self,
            queue: str,
            unique_history: bool = False,
            soft_capacity=None,
            redis_client=None,
        ):
            if not _has_redis:
                raise RuntimeError("Please install the python module: redis")

            self.queue = queue
            self.unique_history = unique_history
            if self.unique_history:
                self.history_queue = self.queue + "_all"
            if redis_client is None:
                redis_client = redis.StrictRedis(
                    host="localhost", port=6379, db=0, decode_responses=True
                )
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


if _has_pika:

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

        def __init__(
            self, queue: str, prefetch: int, host: str = "localhost", port: int = 5672
        ):
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
            self.conn = AsyncioConnection(
                pika.ConnectionParameters(host=self.host, port=self.port),
                self.on_connect,
                custom_ioloop=loop,
            )
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
            for i in range(max_batch_size - 1):
                try:
                    l.append(self.buf.get(block=False))
                except queue.Empty:  # disable:
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
            # self.conn.loop.call_soon_threadsafe(self._ack, args=(ack_tags,))
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
            self.conn.ioloop.call_soon_threadsafe(self._ack_upto, ack_tag)

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

        def __init__(
            self,
            queue: str,
            host: str = "localhost",
            port: int = 5672,
            channel: BlockingChannel = None,
        ):
            if not _has_pika:
                raise RuntimeError("Please install the python module: pika")

            if channel is None:
                self.conn = BlockingConnection(
                    pika.ConnectionParameters(host=host, port=port)
                )
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
                "",
                self.queue,
                msg,
                properties=pika.BasicProperties(
                    content_type="text/plain", delivery_mode=2
                ),  # persistent
                mandatory=True,
            )

        def close(self):
            """
            Close the connection.
            After the call to this method, you cannot `send`.
            """
            if self.conn is not None:
                self.conn.close()


if _has_kafka:

    class KafkaProducer(object):
        """
        Args:
            topic (str): the name of a topic (queue) you publish to.
            host (str, optional): Defaults to 'localhost'.
        """

        def __init__(self, topic: str, host: str = "localhost"):
            if not _has_kafka:
                raise RuntimeError("Please install the python module: confluent_kafka")

            self.topic = topic
            self.producer = Producer(
                {
                    "bootstrap.servers": host,
                    "queue.buffering.max.messages": 10000000,
                }  # 1e7
            )

        def send(self, msg: str):
            """send a single string.

            Args:
                msg (str): a message to send.
            """

            self.producer.poll(0)
            self.producer.produce(self.topic, msg.encode(), callback=None)

        def flush(self):
            """flush the reamining kafka messages.
            """

            self.producer.flush()

    class KafkaConsumer(object):
        def __init__(
            self, topic: str, prefetch: int, consumer_id: str, host: str = "localhost"
        ):
            """
            Args:
                topic (str): the name of a topic.
                consumer_id (str): Kafka remembers the last message read by consumer_id.
                host (str, optional): Defaults to 'localhost'. [description]
            """

            if not _has_kafka:
                raise RuntimeError("Please install the python module: confluent_kafka")

            self.topic = topic
            self.host = host
            self.consumer_id = consumer_id
            self.running = False

            self.m = mp.Manager()
            self.q = self.m.Queue(prefetch)

            self.running = False
            self.start_drain()

        def get(self, max_batch_size: int) -> List[str]:
            return pop(self.q, max_batch_size)

        def iter(self, max_batch_size: int):
            while True:
                yield pop(self.q, max_batch_size)

        @staticmethod
        def _drain(host, consumer_id, topic, q):
            consumer = Consumer(
                {
                    "bootstrap.servers": host,
                    "group.id": consumer_id,
                    "default.topic.config": {"auto.offset.reset": "smallest"},
                }
            )
            consumer.subscribe([topic])
            while True:
                msg = consumer.poll(1.0)
                if (msg is not None) and (msg.error() is None):
                    q.put(msg.value().decode(), block=True)

        def start_drain(self):
            if not self.running:
                self.running = True
                self.p = Process(
                    target=self._drain,
                    args=(self.host, self.consumer_id, self.topic, self.q),
                )
                self.p.start()

        def stop_drain(self):
            if self.running:
                self.running = False
                raise NotImplementedError()


if _has_nats:

    class NatsConsumer(object):
        """A Nats Streaming consumer using durable queues.

        It allows you to resume your progress with manual acks.

        A durable subscription is identified by durable_name & client_name.

        Args:
            subject (str) : a subject to subscribe
            max_flight (int) : the maximum number of msgs to hold in the client
            durable_name (str)
            client_name (str) : Defaults to 'clientid'.
            cluster_id (str) : Defaults to 'test-cluster'.
        """

        def __init__(
            self,
            subject: str,
            max_inflight: int,
            client_name: str,
            durable_name: str = "durable",
            cluster_id: str = "test-cluster",
        ):
            if not _has_nats:
                raise RuntimeError("Please install the python module: nats")

            self.subject = subject
            self.max_inflight = max_inflight
            self.durable_name = durable_name
            self.client_name = client_name
            self.cluster_id = cluster_id

            # self.q = mp.Manager().Queue()
            self.q = mp.Queue()
            self.loop = asyncio.new_event_loop()
            self.p = Thread(target=self._run, args=())
            self.p.start()

        def _run(self):
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect())
            self.loop.run_forever()

        async def cb(self, msg):
            # stan.aio.client.Msg = {proto, sub(subscription)}
            # Msg is not pickable
            self.q.put((msg.proto.data.decode(), msg.proto.sequence))
            # self._ack(msg.proto.sequence)

        async def _connect(self):
            self.nc = NATS()
            await self.nc.connect(io_loop=self.loop)

            self.sc = STAN()
            await self.sc.connect(self.cluster_id, self.client_name, nats=self.nc)
            self.sub = await self.sc.subscribe(
                self.subject,
                durable_name=self.durable_name,
                cb=self.cb,
                max_inflight=self.max_inflight,
                start_at="first",
                manual_acks=True,
                ack_wait=30,
            )

        def ack(self, seq: int):
            self.loop.call_soon_threadsafe(self._ack, seq)

        def _ack(self, seq: int):
            if not (isinstance(seq, list) or isinstance(seq, tuple)):
                seq = [seq]

            for seq in seq:
                ack = protocol.Ack()
                ack.subject = self.subject
                ack.sequence = seq
                asyncio.ensure_future(
                    self.sc._nc.publish(self.sub.ack_inbox, ack.SerializeToString()),
                    loop=self.loop,
                )

        async def _close(self):
            await self.sub.unsubscribe()
            await self.sc.close()
            await self.nc.close()

        def close(self):
            raise NotImplementedError
            # asyncio.run_coroutine_threadsafe(self._close(), loop=self.loop)
            # self.loop.call_soon_threadsafe(self.loop.stop)

    class NatsProducer(object):
        """A Nats Streaming producer using durable queues.

        It allows you to resume your progress with manual acks.

        A durable subscription is identified by durable_name & client_name.

        Args:
            subject (str) : a subject to subscribe
            max_flight (int) : the maximum number of msgs to hold in the client
            durable_name (str)
            client_name (str) : Defaults to 'clientid'.
            cluster_id (str) : Defaults to 'test-cluster'.
        """

        def __init__(
            self, subject: str, client_name: str, cluster_id: str = "test-cluster"
        ):
            if not _has_nats:
                raise RuntimeError("Please install the python module: nats")

            self.wg = WaitGroup()
            self.startup_lock = mp.Lock()

            self.subject = subject
            self.client_name = client_name
            self.cluster_id = cluster_id

            self.startup_lock.acquire()
            self.loop = asyncio.new_event_loop()
            self.p = Thread(target=self._run)
            self.p.start()

            self.startup_lock.acquire()
            atexit.register(self.close)

            self.closed = False

        def _run(self):
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect())
            self.startup_lock.release()
            self.loop.run_forever()

        async def _connect(self):
            self.nc = NATS()
            await self.nc.connect(io_loop=self.loop)

            self.sc = STAN()
            await self.sc.connect(self.cluster_id, self.client_name, nats=self.nc)

        def send(self, msg: str):
            self.wg.add(1)
            self.loop.call_soon_threadsafe(self._send, msg)

        def _send(self, msg: str):
            asyncio.ensure_future(
                self.sc.publish(
                    self.subject, msg.encode(), ack_handler=self.ack_handler
                )
            )

        async def ack_handler(self, _ack):
            self.wg.done()

        async def _close(self):
            await self.sc.close()
            await self.nc.close()

        def flush(self):
            self.wg.wait()

        def close(self):
            raise NotImplementedError
            # if not self.closed:
            #     self.closed = True
            #     self.flush()
            #     asyncio.run_coroutine_threadsafe(self._close(), loop=self.loop)
            #     self.loop.call_soon_threadsafe(self.loop.stop)
