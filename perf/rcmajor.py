#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

from bettertimeit import bettertimeit


def forward():
    import numpy as np
    N = 3000
    indim = 20
    hiddendim = 40

    w = np.random.randn(indim, hiddendim)
    x = np.random.randn(N, indim)

    def timeit_rowmajor():
        x @ w

    wopp = np.random.randn(hiddendim, indim)
    xopp = np.random.randn(indim, N)

    def timeit_colmajor():
        wopp @ xopp

    def timeit_blendr():
        x @ wopp.T

    def timeit_blendc():
        w.T @ xopp


bettertimeit(forward, 10)

