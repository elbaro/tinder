import tinder
import time

p = tinder.queue.NatsProducer('topic')

i = 0
while True:
    i += 1
    msg = 'hi ' + str(i)
    print(msg)
    p.send(msg)
    time.sleep(1)

