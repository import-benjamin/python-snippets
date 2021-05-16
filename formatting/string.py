from timeit import timeit
from shlex import join as shjoin


def toIdentifier(_key: str) -> str:
    from re import split

    key_buffer = split(r"[^A-Za-z0-9]+", _key)
    res_buffer = "".join(tuple(map(lambda s: s.title(), key_buffer)))
    return f"{res_buffer[0].lower()}{res_buffer[1:]}"


my_dict = "@branch-rate"


timeit(
    "toIdentifier(my_dict)",
    setup="from __main__ import toIdentifier, my_dict",
    number=1000,
)
# 0.007201906002592295


def string_join():
    return "".join(["this", "is", "a", "string", "jointure", "comparison"])


timeit("string_join()", setup="from __main__ import string_join", number=1000)
# 0.0005619599999135971


def shlex_join():
    return shjoin(["this", "is", "a", "string", "jointure", "comparison"])


timeit("shlex_join()", setup="from __main__ import shlex_join", number=1000)
# Don't have installed python 3.8 for now.
