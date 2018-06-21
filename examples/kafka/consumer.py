import tinder
import time

q = tinder.serving.KafkaConsumer('test_q', 'example_consumer')

for batch in q.iter(100):
    print(batch)
    time.sleep(1)
