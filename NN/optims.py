#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"


class SGDReference:
    """
    SGD is defined as

    new_params = current_params - learning_rate * gradient
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, param, gradient):
        return param - self.lr * gradient


class MomentumGDReference:
    """
    Momentum Gradient Descent remembers previous gradients

    moment = momentum * previous_moment + learning_rate * gradient
    new_params = current_params - moment

    Note:
        In this case, the optimiser must store state. This is because the momentum
        equations are recursive and would require the entire gradient history
        to reconstruct.

        Since the momentum equations change for each optimiser, we have to let the
        optimisers hold state.
    """

    def __init__(self, momentum=0.9, lr=0.01):
        self.momentum = momentum
        self.lr = lr
        self.moment = 0

    def step(self, param, gradient):
        moment = self.momentum * self.moment + self.lr * gradient
        self.moment = moment
        return param - moment
