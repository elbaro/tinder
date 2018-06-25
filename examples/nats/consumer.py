import tinder

c = tinder.queue.NatsConsumer('topic', max_inflight=5, client_name='example_client')

while True:
    for batch, ack in tinder.batch.iter_with_ack_from_q(c.q, 3):
        print('batch:', batch, 'ack:', ack)
        c.ack(ack)
