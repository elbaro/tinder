import tinder
import time

p = tinder.queue.NatsProducer('topic', client_name='example_producer')

i = 0
while True:
    i += 1
    msg = 'hi ' + str(i)
    print(msg)
    p.send(msg)
    time.sleep(1)

