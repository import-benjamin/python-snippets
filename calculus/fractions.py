from fractions import Fraction
from timeit import timeit


def python_fraction():
    return Fraction(22, 7)


timeit("python_fraction()", setup="from __main__ import python_fraction", number=1000)
# 0.002394800000004693


def frac_operator():
    return 22 / 7


timeit("frac_operator()", setup="from __main__ import frac_operator", number=1000)
# 0.00010830800000860563
