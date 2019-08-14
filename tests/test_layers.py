#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np


def test_fc():
    from NN.layers import FullyConnected
    l1 = FullyConnected(5, 10)
    x = np.ones((100, 5))
    y = l1.forward(x)
    assert y.shape == (100, 10)


def test_loss():
    from NN.layers import MSELoss
    y = np.random.randn(100, 10)
    yhat = np.random.randn(100, 10)
    mse = MSELoss()
    assert mse.loss(y, yhat) > 0
    assert type(mse.loss_gradient(y, yhat)) == np.float64
