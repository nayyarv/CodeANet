#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np


class Layer:
    def forward(self, x):
        pass

    def backward(self, dldy, cache):
        pass


class FullyConnected(Layer):
    def __init__(self, indim, hiddendim):
        super().__init__()
        self.W = np.random.randn(indim, hiddendim) * np.sqrt(1 / indim)
        self.b = np.zeros(hiddendim)

    def forward(self, x):
        y = x @ self.W + self.b
        return y, x

    def backward(self, dldy, x):
        dldw = x.T @ dldy
        dldb = np.sum(dldy, axis=0)
        dldx = dldy @ self.W.T

        self.W -= dldw * LR
        self.b -= dldb * LR

        return dldx


class Tanh(Layer):

    def forward(self, x):
        y = np.tanh(x)
        return y, y

    def backward(self, dldy, y):
        return (1 - y ** 2) * dldy


class Network(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.network = tuple(layers)

    def forward(self, x):
        cacheList = []
        for l in self.network:
            x, c = l.forward(x)
            cacheList.append(c)
        return x, cacheList

    def backward(self, dldx, cachelist):
        for l, c in zip(reversed(self.network), reversed(cachelist)):
            dldx = l.backward(dldx, c)
        return dldx


def train(network, data, numepochs):
    from NN.loss import MSELoss
    mse = MSELoss()
    for i in range(numepochs):
        x, y = data
        yhat, cachelist = network.forward(x)
        dldy = mse.loss_gradient(y, yhat)
        network.backward(dldy, cachelist)
        print(f"Epoch {i}, loss: {mse.loss(y, yhat)}")
