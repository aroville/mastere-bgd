# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:06:31 2016

@author: claire
"""
import numpy as np
from random import random, randint

# from random import betavariate
from math import log


def epsilon_greedy(n_arms, epsilon, rewards, draws):
    if np.sum(draws == 0) > 0:
            c = np.where(draws == 0)[0][0]
    else:
        u = random()
        if u < epsilon:
            c = randint(0, n_arms - 1)
        else:
            indices = rewards / draws
            winners = np.argwhere(indices == np.max(indices))
            c = np.random.choice(winners[0])
    return c


def ucb(t, alpha, rewards, draws):
    if np.sum(draws == 0) > 0:
        c = np.where(draws == 0)[0][0]
    else:
        indices = (rewards/draws) + np.sqrt(alpha*np.log(t) / (2*draws))
        winners = np.argwhere(indices == np.max(indices))
        c = np.random.choice(winners[0])
    return c


def thompson(n_arms, rewards, draws):
    alpha, beta = 1, 1

    indices = np.zeros(n_arms)
    for arm in np.arange(n_arms):
        indices[arm] = rewards / draws

    winners = np.argwhere(indices == np.max(indices))
    c = np.random.choice(winners[0])
    return c


def kl(a, b):
    return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))


def compute_lower_bound(n_arms, true_means):
    mu_1 = np.max(true_means)
    return np.sum([(mu_1-mu_k)/kl(mu_k, mu_1)
                   for mu_k in true_means
                   if mu_k != mu_1])
