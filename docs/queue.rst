
Queue
===================================

These queues require additional dependencies.

========  ======
Queue     Python Dependency
========  ======
Redis     redis
RabbitMQ   pika
Kafka     confluent_kafka
Nats      nats
========  ======


.. currentmodule:: tinder.queue

.. autoclass:: NatsConsumer
    :members:

.. autoclass:: NatsProducer
    :members:

.. autoclass:: KafkaConsumer
    :members:

.. autoclass:: KafkaProducer
    :members:

.. autoclass:: RabbitConsumer
    :members:

.. autoclass:: RabbitProducer
    :members:

.. autoclass:: RedisQueue
    :members:
