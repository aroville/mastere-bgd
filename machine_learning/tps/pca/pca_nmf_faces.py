#  License: BSD
#  -*- coding: utf-8 -*-

#  Authors: Vlad Niculae, Alexandre Gramfort, Slim Essid


from time import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_olivetti_faces
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# -- Prepare data and define utility functions --------------------------------

n_row, n_col = 2, 5
n_components = 35 #n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape


print("Dataset consists of %d faces" % n_samples)


def plot_gallery(title, images):
    """Plot images as gallery"""
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        comp = comp.reshape(image_shape)
        vmax = comp.max()
        vmin = comp.min()
        dmy = np.nonzero(comp < 0)
        comp[comp < 0] = 0

        plt.imshow(comp, cmap=plt.cm.gray, vmax=vmax, vmin=vmin)

        if len(dmy[0]) > 0:
            yz, xz = dmy
            plt.plot(xz, yz, 'r,', hold=True)
            print(len(dmy[0]), "negative-valued pixels")

        plt.xticks(())
        plt.yticks(())

    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

# global centering
faces_centered = faces - faces.mean(axis=0, dtype=np.float64)

# Plot a sample of the input data
# plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# -- Decomposition methods ----------------------------------------------------

# List of the different estimators and whether to center the data
estimators = [
    (
        'pca',
        'Eigenfaces - PCA',
        PCA(n_components=n_components, whiten=True),
        True),
    (
        'nmf',
        'Non-negative components - NMF',
        NMF(n_components=n_components,
            init=None, tol=1e-6, sparseness=None, max_iter=1000),
        False)
]

# -- Transform and classify ---------------------------------------------------

labels = dataset.target
X = faces
X_ = faces_centered


scores = {'pca': [], 'nmf': []}
for shortname, name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    data = estimator.fit_transform(X_ if center else X)
    score = np.mean(cross_val_score(LinearDiscriminantAnalysis(), data, labels, cv=8, n_jobs=-1))
    scores[shortname].append(score)

#     plot_gallery('%s - Train time %.1fs' % (name, train_time),
#                  estimator.components_[:n_components])
#
#
# plt.show()