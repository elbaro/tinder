import tinder
import time

q = tinder.serving.RabbitConsumer('test_q')

for (batch,ack) in q.iter_batch(100):
    assert len(batch) == len(ack)
    print(batch)
    q.ack(ack)
    time.sleep(1)
