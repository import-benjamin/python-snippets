from timeit import timeit


def new_dict(input: dict) -> dict:
    new_dict = dict()
    for key in input.keys():
        new_dict.update({key: input.get(key)})
    return new_dict


def update_dict(input: dict) -> dict:
    for key in input.keys():
        input.update({key: input.pop(key)})
    return input


def unpack_dict(input: dict) -> dict:
    for key in input.keys():
        input = {**input, key: input.get(key)}
    return input


def unpack_dict_inline(input: dict) -> dict:
    input = {**input, **{key: input.get(key) for key in input.keys()}}
    return input


def unpack_new_dict_inline(input: dict) -> dict:
    new_dict = dict()
    new_dict = {key: input.get(key) for key in input.keys()}
    return new_dict


# Create new dictionnary with 1000 items
my_dict = {f"{k}": k for k in range(1000)}
# {'198': 198, ..., '280': 280, '28...

timeit("new_dict(my_dict)", setup="from __main__ import new_dict, my_dict", number=100)
# 0.029584372000044823


timeit(
    "update_dict(my_dict)",
    setup="from __main__ import update_dict, my_dict",
    number=100,
)
# 0.08109318400011034

timeit(
    "unpack_dict(my_dict)",
    setup="from __main__ import unpack_dict, my_dict",
    number=100,
)
# 2.406963264000069

timeit(
    "unpack_dict_inline(my_dict)",
    setup="from __main__ import unpack_dict_inline, my_dict",
    number=100,
)
# 0.025858885000161536

timeit(
    "unpack_new_dict_inline(my_dict)",
    setup="from __main__ import unpack_new_dict_inline, my_dict",
    number=100,
)
# 0.019454515999996147
