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
        print('.')
        batch = consumer.consume(num_messages=100, timeout=5)
        if len(batch) > 0:
            break
    print(len(batch))
    print('---')
    time.sleep(1)
