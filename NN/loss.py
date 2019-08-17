#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np


class MSELoss:
    def loss(self, y, yhat):
        return np.mean((y - yhat)**2 / 2)

    def loss_gradient(self, y, yhat):
        return np.expand_dims(np.mean(yhat - y, axis=-1), axis=-1)