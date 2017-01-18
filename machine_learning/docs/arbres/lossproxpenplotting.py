# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:49:12 2016

@author: salmon
"""

# from functools import partial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # for plots
from matplotlib import rc
from prox_collections import loss_square, loss_logistic, loss_hinge, loss_01

###############################################################################
# Plot initialization

plt.close('all')
dirname = "../srcimages/"
imageformat = '.pdf'


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 16,
          'legend.fontsize': 16,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
# sns.set_style("ticks")
# sns.axes_style()

is_prox_needed = False


def my_saving_display(fig, dirname, filename, imageformat):
    """"saving faster"""
    dirname + filename + imageformat
    image_name = dirname + filename + imageformat
    fig.savefig(image_name)

###############################################################################
# ploting prox operators and penalty functions


def plot_prox(x, threshold, prox, label, image_name):
    """ function to plot and save prox operators"""
    z = np.zeros(x.shape)
    for i, value in enumerate(np.nditer(x)):
        z[i] = prox(value, threshold)

    fig0 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    ax1.plot(x, z, label=label)
    # ax1.plot(x, log_prox(x, 3, 1),
    #          label=r"$\eta_{\rm {log},\lambda,\gamma}$")
    ax1.plot(x, x, 'k--', linewidth=1)
    plt.legend(loc="upper left", fontsize=34)
    # ax1.get_yaxis().set_ticks([])
    # ax1.get_xaxis().set_ticks([])
    plt.show()
    my_saving_display(fig0, dirname, image_name, imageformat)
    return fig0


def plot_loss(x, threshold, pen, image_name):
    """ function to plot and save pen functions"""
    xx = pen(x, threshold)
    fig0 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(111)
    ax1.plot(x, xx, label=label)
    # plt.title(label, fontsize=24)
    # plt.subplots_adjust(top=0.80)
    # ax1.get_yaxis().set_ticks([])
    # ax1.get_xaxis().set_ticks([])
    ax1.set_ylim(-0.5, 5)
    ax1.set_xlim(-2, 3)
    plt.show()
    my_saving_display(fig0, dirname, image_name, imageformat)
    return fig0


x = np.arange(-10, 10, step=0.01)


# l01 loss
image_name = "l01_loss"
label = r"$\eta_{0}$"
loss = loss_01
plot_loss(x, 0, loss, image_name)

# hinge loss
image_name = "hinge_loss"
label = r"$\eta_{0}$"
loss = loss_hinge
plot_loss(x, 1, loss, image_name)

# logistic loss
image_name = "logistic_loss"
label = r"$\eta_{0}$"
loss = loss_logistic
plot_loss(x, 2, loss, image_name)

# square loss
image_name = "square_loss"
label = r"$\eta_{0}$"
loss = loss_square
plot_loss(x, 1, loss, image_name)

###############################################################################
# ploting loss functions altogether

fig0 = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(111)

ax1.plot(x, loss_01(x, 0), label='l01 loss')
ax1.plot(x, loss_square(x, 1), label='Square loss')
ax1.plot(x, loss_hinge(x, 1), label='Hinge loss')
ax1.plot(x, loss_logistic(x, 2), label='Logistics loss')
ax1.set_ylim(-0.5, 5)
ax1.set_xlim(-2, 3)
# ax1.get_yaxis().set_ticks([])
# ax1.get_xaxis().set_ticks([])
plt.legend(loc="upper center", fontsize=14)

plt.show()
my_saving_display(fig0, dirname, "losses", imageformat)
