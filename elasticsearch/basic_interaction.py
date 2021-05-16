#!/usr/bin/env python

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import os

# to connect without credentials
# define your ELK IP into an .env file -> ELK_IP=0.0.0.0

if __name__ == "__main__":
    # Connect to the elasticsearch
    es = Elasticsearch([os.getenv("ELK_IP")])
    print(es.info())

    # Index event
    es.index(index="demo", body={"event": {"message": "Hello world!"}})
    es.indices.refresh(index="demo")

    # event.message turn into event__message <- double underscore
    request = Q("match_phrase", event__message="Hello world!")
    search = Search(using=es).query(request)

    # result contains elastic common json structure ... hits.hits => []
    result = search.execute()
    print(result.hits.hits)
    print(result.hits.hits[0]._source.event.message)

    # Clean
    for hit in result.hits.hits:
        # Remove entry by _id
        es.delete(index="demo", id=hit._id)
