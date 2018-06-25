import time
from confluent_kafka import Producer

producer = Producer({
    'bootstrap.servers': 'localhost',
    'queue.buffering.max.messages': 10000000,  # 1e7
})

i = 0

while True:
    i += 1
    print(i)

    msg = 'msg ' + str(i)
    producer.poll(0)
    producer.produce('test_q', msg.encode(), callback=None)
    #time.sleep(0.1)

    if i==20:
        break

time.sleep(10)
