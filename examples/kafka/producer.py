import tinder
import time

q = tinder.queue.KafkaProducer("test_q")
i = 0

while True:
    i += 1
    print(i)
    q.send("msg " + str(i))
    time.sleep(0.1)
