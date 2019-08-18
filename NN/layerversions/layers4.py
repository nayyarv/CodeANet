#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
import functools


def sgd_step(lr, grad):
    return lr * grad


def sgd_optimiser(lr):
    return functools.partial(sgd_step, lr)


class Layer:
    def forward(self, x):
        pass

    def backward(self, dldy, cache, optimiser):
        pass

    def params(self):
        pass


class FullyConnected(Layer):
    def __init__(self, indim, hiddendim):
        super().__init__()
        self.W = np.random.randn(indim, hiddendim) * np.sqrt(2 / indim)
        self.b = np.zeros(hiddendim)

    def params(self):
        return self.W, self.b

    def forward(self, x):
        y = x @ self.W + self.b
        return y, x

    def backward(self, dldy, x, optimiser):
        dldw = x.T @ dldy
        dldb = np.sum(dldy, axis=0)
        dldx = dldy @ self.W.T

        self.W -= optimiser(dldw)
        self.b -= optimiser(dldb)

        return dldx


class Tanh(Layer):

    def params(self):
        return ()

    def forward(self, x):
        y = self.forward(x)
        return y, y

    def backward(self, dldy, y, _):
        return (1 - y ** 2) * dldy


class Network(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.network = tuple(layers)

    def params(self):
        return tuple(l.params() for l in self.network)

    def forward(self, x):
        for l in self.network:
            x, _ = l.forward(x)
        return x

    def forward_train(self, x):
        cache_list = []
        for l in self.network:
            x, c = l.forward_train(x)
            cache_list.append(c)
        return x, cache_list

    def backward(self, dldx, cache_list, optimiser):
        for l, c in zip(reversed(self.network), reversed(cache_list)):
            dldx = l.backward(dldx, c, optimiser)
        return dldx


def train(network, data, numepochs, loss=None, optim=None):
    if not loss:
        from NN.loss import MSELoss
        loss = MSELoss()
    if not optim:
        optim = sgd_optimiser(0.01)
    for i in range(numepochs):
        x, y = data
        yhat, cachelist = network.forward(x)
        dldy = loss.loss_gradient(y, yhat)
        network.backward(dldy, cachelist, optim)
        print(f"Epoch {i}, loss: {loss.loss(y, yhat)}")
