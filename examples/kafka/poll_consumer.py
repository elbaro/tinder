from confluent_kafka import Consumer, KafkaError

consumer = Consumer({
    'bootstrap.servers': 'localhost',
    'group.id': 'example_consumer',
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
     }
})

consumer.subscribe(['test_q'])

while True:
    msg = consumer.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue
        else:
            print(msg.error())
            break

    print('{}'.format(msg.value().decode('utf-8')))

c.close()
