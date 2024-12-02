from math import sin,cos,sqrt,exp,log,pi,factorial
import numpy as np
import random
from itertools import * 

def permutation_index(element, iterable):
    index = 0
    pool = list(iterable)
    for i, x in zip(range(len(pool), -1, -1), element):
        r = pool.index(x)
        index = index * i + r
        del pool[r]
    return index

def nth_permutation(iterable, r, index):
    pool = list(iterable)
    n = len(pool)
    if r is None or r == n:
        r, c = n, factorial(n)
    elif not 0 <= r < n:
        raise ValueError
    else:
        c = factorial(n) // factorial(n - r)
    if index < 0:
        index += c
    if not 0 <= index < c:
        raise IndexError
    if c == 0:
        return tuple()
    result = [0] * r
    q = index * factorial(n) // c if r < n else index
    for d in range(1, n + 1):
        q, i = divmod(q, d)
        if 0 <= n - d < r:
            result[n - d] = i
        if q == 0:
            break
    return tuple(map(pool.pop, result))

def perm_mult(g1,g2):
    "Multiplies two permutations g1 and g2"
    gv=np.array(g1)
    return gv[np.array(g2[:])]

def perm_inv(g1):
    "constructs the inverse of a permutations g1"
    return np.array([np.where(np.array(g1)==i)[0][0] for i in range(len(g1))])