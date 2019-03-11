import tinder


def preprocess(msg: str):
    return msg + str(len(msg))


c = tinder.queue.KafkaConsumer(topic="sfd", consumer_id="anonymous_123")

print("consumer.get: ", c.get(max_batch_size=3))

q = c.start_drain(batch_size=3, capacity=20)
loader = tinder.dataset.StreamingDataloader(
    q, batch_size=5, num_workers=2, transform=preprocess
)

for batch in loader:
    print("batch: ", batch)
