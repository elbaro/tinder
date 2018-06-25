import time
from confluent_kafka import Consumer

consumer = Consumer({
    'bootstrap.servers': 'localhost',
    'group.id': 'example_consumer',
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
    }
})

consumer.subscribe(['test_q'])

while True:
    while True:
        t0 = time.time()
        batch = consumer.consume(num_messages=100, timeout=1)
        t = time.time()
        print('consume() took ', t-t0)
        if len(batch) > 0:
            break
    batch = [sample.value().decode() for sample in batch if (sample.error() is None)]
    print(len(batch), batch)
    #time.sleep(1)
