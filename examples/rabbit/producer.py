import tinder
import time

q = tinder.serving.RabbitProducer('test_q')

i = 0
while True:
    i += 1
    print(i)
    q.send('random ' + str(i))
    time.sleep(3)
