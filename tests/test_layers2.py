#!/usr/bin/env py.test
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np

import NN.layerversions.layers2 as layer


def test_fc():
    l1 = layer.FullyConnected(5, 10)
    x = np.ones((100, 5))
    cachedict = {}
    y = l1.forward(x, cachedict)
    assert y.shape == (100, 10)
    assert len(cachedict) == 1
    # check for equality here!
    assert np.all(list(cachedict.values())[0] == x)


def test_tanh():
    l = layer.Tanh()
    x = np.ones((100, 5))
    cachedict = {}
    y = l.forward(x, cachedict)
    assert y.shape == (100, 5)
    assert len(cachedict) == 1
    # check for equality here!
    assert np.all(list(cachedict.values())[0] == y)


def test_back_fc():
    l1 = layer.FullyConnected(5, 10)
    x = np.ones((100, 5))
    dldy = np.random.randn(100, 10)

    dldx = l1.backward(dldy, {l1.name: x})
    assert dldx.shape == (100, 5)


def test_back_tanh():
    l1 = layer.Tanh()
    x = np.random.randn(100, 5)
    dldy = np.random.randn(100, 5)

    dldx = l1.backward(dldy, {l1.name: np.tanh(x)})
    assert dldx.shape == (100, 5)


def test_network():
    from NN.loss import MSELoss
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 3)
    net = layer.Network(
        layer.FullyConnected(10, 20),
        layer.Tanh(),
        layer.FullyConnected(20, 3),
        layer.Tanh()
    )
    mse = MSELoss()
    yhat = net.forward(x, {})
    initloss = mse.loss(y, yhat)
    layer.train(net, (x, y), 10)
    yhat = net.forward(x, {})
    finloss = mse.loss(yhat, y)

    assert initloss > finloss

