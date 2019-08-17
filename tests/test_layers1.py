#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np

import NN.layerversions.layers1 as layer


# @pytest.fixture(params=[1, 2, 3])
# def layer(request):
#     fl = f"NN.layers{request.param}"
#     mod = importlib.import_module(fl)
#     return mod

def test_fc():
    l1 = layer.FullyConnected(5, 10)
    x = np.ones((100, 5))
    y = l1.forward(x)
    assert y.shape == (100, 10)


def test_tanh():
    l = layer.Tanh()
    x = np.ones((100, 5))
    y = l.forward(x)
    assert y.shape == (100, 5)
