#!/usr/bin/env py.test
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import pytest


def test_loss():
    from NN.loss import MSELoss
    y = np.random.randn(100, 10)
    yhat = np.random.randn(100, 10)
    mse = MSELoss()
    assert mse.loss(y, yhat) > 0
    assert mse.loss_gradient(y, yhat).shape == (100, 1)