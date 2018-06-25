import tinder

c = tinder.queue.NatsConsumer('topic', max_inflight=5, durable_name='d3')


while True:
    batch = tinder.batch.pop(c.q,3)
    print(batch)

