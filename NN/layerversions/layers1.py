#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np


class Layer:
    def forward(self, x):
        pass

    def backward(self, dldy):
        pass


class FullyConnected(Layer):
    def __init__(self, indim, hiddendim, init="xavier"):
        if init == "xavier":
            scale = np.sqrt(2/(indim+hiddendim))
        elif init == "he":
            scale = np.sqrt(2/indim)
        else:
            raise ValueError(f"Unknown initialiser: {init}")
        self.W = np.random.randn(indim, hiddendim) * scale
        self.b = np.zeros(hiddendim)


    def forward(self, x):
        y = x @ self.W + self.b
        return y

    def backward(self, dldy):
        dldw = x.T @ dldy
        dldb = np.sum(dldy, axis=0)
        dldx = dldy @ self.W.T

        return dldx


class Tanh(Layer):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, dldy):
        dldx = (1 - (np.tanh(x)) ** 2) * dldy
        return dldx


class Network(Layer):
    def __init__(self, *layers):
        self.network = tuple(layers)

    def forward(self, x):
        for l in self.network:
            x = l.forward(x)
        return x

    def backward(self, dldx):
        for l in reversed(self.network):
            dldx = l.backward(dldx)
        return dldx

    def train(self, data, numepochs):
        from NN.loss import MSELoss
        mse = MSELoss()
        for i in range(numepochs):
            x, y = data
            yhat = self.forward(x)
            dldy = mse.loss_gradient(y, yhat)
            self.backward(dldy)
            print(f"Epoch {i}, loss: {mse.loss(y, yhat)}")
