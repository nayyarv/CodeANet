#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
from uuid import uuid1

LR = 1e-4


class Layer:
    def __init__(self):
        self.name = uuid1()

    def forward(self, x, cachedict):
        pass

    def backward(self, dldy, cachedict):
        pass


class FullyConnected(Layer):
    def __init__(self, indim, hiddendim):
        super().__init__()
        self.W = np.random.randn(indim, hiddendim) * np.sqrt(2 / indim)
        self.b = np.zeros(hiddendim)
        self.name = uuid1()

    def forward(self, x, cachedict):
        y = x @ self.W + self.b
        cachedict[self.name] = x
        return y

    def backward(self, dldy, cachedict):
        x = cachedict[self.name]
        dldw = x.T @ dldy
        dldb = np.sum(dldy, axis=0)
        dldx = dldy @ self.W.T

        self.W -= dldw * LR
        self.b -= dldb * LR

        return dldx


class Tanh(Layer):

    def forward(self, x, cachedict):
        th = np.tanh(x)
        cachedict[self.name] = th
        return th

    def backward(self, dldy, cachedict):
        th = cachedict[self.name]
        return (1 - th ** 2) * dldy


class Network(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.network = tuple(layers)

    def forward(self, x, cachedict):
        for l in self.network:
            x = l.forward(x, cachedict)
        return x

    def backward(self, dldx, cachedict):
        for l in reversed(self.network):
            dldx = l.backward(dldx, cachedict)
        return dldx


def train(network, data, numepochs):
    from NN.loss import MSELoss
    mse = MSELoss()
    for i in range(numepochs):
        cachedict = {}
        x, y = data
        yhat = network.forward(x, cachedict)
        dldy = mse.loss_gradient(y, yhat)
        network.backward(dldy, cachedict)
        print(f"Epoch {i}, loss: {mse.loss(y, yhat)}")
