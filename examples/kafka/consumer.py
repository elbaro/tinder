import tinder
import time

q = tinder.queue.KafkaConsumer("test_q", 10, "example_consumer")

for batch in q.iter(10):
    print(batch)
    # time.sleep(1)
