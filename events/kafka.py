from kafka import KafkaConsumer
from kafka import KafkaProducer
from json import dumps, loads
from pprint import pprint


server = "<my.server.com>:<port_number>"


def main():
    producer = KafkaProducer(
        bootstrap_servers=server,
        security_protocol='SSL',
        ssl_cafile='<CARoot>',
        ssl_certfile='<certificate>',
        ssl_keyfile='<key>',
        value_serializer=lambda v: dumps(v).encode("utf-8"))

    consumer = KafkaConsumer(
        "my_topic",
        bootstrap_servers=server,
        security_protocol='SSL',
        ssl_cafile='<CARoot>',
        ssl_certfile='<certificate>',
        ssl_keyfile='<key>',
        value_deserializer=lambda v: loads(v))

    producer_metrics, consumer_metrics = producer.metrics(), consumer.metrics()
    pprint(producer_metrics)
    pprint(consumer_metrics)

    for msg in consumer:
        print(msg)
        producer.send("new_topic", msg.value)
        producer.flush()


if __name__ == '__main__':
    main()
