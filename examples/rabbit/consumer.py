import tinder
import time

q = tinder.queue.RabbitConsumer("test_q", prefetch=200)

for (batch, ack) in q.iter_batch(100):
    assert len(batch) == len(ack)
    print(batch)
    q.ack(ack)
    time.sleep(1)
