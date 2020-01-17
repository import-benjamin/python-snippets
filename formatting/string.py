from timeit import timeit


def toIdentifier(_key: str) -> str:
    from re import split
    key_buffer = split(r"[^A-Za-z0-9]+", _key)
    res_buffer = ''.join(tuple(map(lambda s: s.title(), key_buffer)))
    return f"{res_buffer[0].lower()}{res_buffer[1:]}"


my_dict = "@branch-rate"


timeit("toIdentifier(my_dict)", setup="from __main__ import toIdentifier, my_dict", number=1000)
# 0.007201906002592295
