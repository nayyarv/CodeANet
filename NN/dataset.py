#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np


def bin_dat(func):
    from itertools import product
    x, y = [], []
    for a, b in product([0, 1], [0, 1]):
        x.append([a, b])
        y.append(func(a, b))
    return np.array(x), np.array(y)


and_dat = bin_dat(lambda a, b: a & b)
xor_dat = bin_dat(lambda a, b: a ^ b)

if __name__ == '__main__':
    print(and_dat)
    print(xor_dat)