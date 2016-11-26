# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:49:12 2014

@author: salmon
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from os import mkdir, path
from mpl_toolkits.mplot3d import Axes3D


# Uncomment the following 2 lines for Mac OS X / Spyder for using Tex display
# import os as macosx
# macosx.environ['PATH'] = macosx.environ['PATH'] + ':/usr/texbin'

###############################################################################
# Plot initialization

dirname = "../srcimages/"
if not path.exists(dirname):
    mkdir(dirname)

imageformat = '.pdf'
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
plt.close("all")

# sns.set_context("poster")
sns.set_palette("colorblind")
sns.axes_style()
sns.set_style({'legend.frameon': True})
color_blind_list = sns.color_palette("colorblind", 8)
my_orange = color_blind_list[2]
my_green = color_blind_list[1]


###############################################################################
# display function:


def my_saving_display(fig, dirname, filename, imageformat):
    """"saving faster"""
    dirname + filename + imageformat
    image_name = dirname + filename + imageformat
    fig.savefig(image_name)

###############################################################################
# 3D case drawing
plt.close("all")

# Load data
url = 'http://vincentarelbundock.github.io/Rdatasets/csv/datasets/trees.csv'
dat3 = pd.read_csv(url)

# Fit regression model
X = dat3[['Girth', 'Height']]
X = sm.add_constant(X)
y = dat3['Volume']
results = sm.OLS(y, X).fit().params


XX = np.arange(8, 22, 0.5)
YY = np.arange(64, 90, 0.5)
xx, yy = np.meshgrid(XX, YY)
zz = results[0] + results[1] * xx + results[2] * yy


fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('Girth')
ax.set_ylabel('Height')
ax.set_zlabel('Volume')
ax.set_zlim(5, 80)
ax.plot(X['Girth'], X['Height'], y, 'o')
plt.show()
my_saving_display(fig, dirname, "tree_data", imageformat)

# ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10, alpha=0.3)
ax.plot_surface(xx, yy, zz, alpha=0.3)
my_saving_display(fig, dirname, "tree_data_plot_regression", imageformat)

###############################################################################
# Non trivial minima

sns.set_style("white")

XX = np.arange(-1, 1, 0.05)
YY = XX
xx, yy = np.meshgrid(XX, YY)
zz = (xx - yy) ** 2


fig = plt.figure()
ax = Axes3D(fig)

ax.view_init(elev=20., azim=50)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

plt.axis('off')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=2,
                       antialiased=False, alpha=0.5)
my_saving_display(fig, dirname, "CN0_2d_non_trivial1", imageformat)

ax.view_init(elev=20., azim=90)
surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=2,
                       antialiased=False, alpha=0.5)
my_saving_display(fig, dirname, "CN0_2d_non_trivial2", imageformat)

ax.view_init(elev=20., azim=130)
surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=2,
                       antialiased=False, alpha=0.5)
my_saving_display(fig, dirname, "CN0_2d_non_trivial3", imageformat)

ax.view_init(elev=20., azim=170)
surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=2,
                       antialiased=False, alpha=0.5)
my_saving_display(fig, dirname, "CN0_2d_non_trivial4", imageformat)

ax.view_init(elev=20., azim=210)
surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=2,
                       antialiased=False, alpha=0.5)
my_saving_display(fig, dirname, "CN0_2d_non_trivial5", imageformat)
