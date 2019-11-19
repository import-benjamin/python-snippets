#!/usr/bin/env python

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, A
import os

# to connect without credentials
# Use export ELK_IP="10.171.40.89"

if __name__ == '__main__':
    es = Elasticsearch([os.getenv("ELK_IP")])
    es.index(index="demo", body={"event": {"message": "Hello world!"}})

    print(es.info())

    # event.message turn into event__message <- double underscore
    request = Q("match_phrase", event__message="Hello world!")
    search = Search(using=es).query(request)

    # result contains elastic common json structure ... hits.hits => []
    result = search.execute()
    print(result.hits.hits)

    # Clean
    for hit in result.hits.hits:
        #print(hit)
        # Remove entry by _id
        es.delete(index="demo", id=hit._id)
