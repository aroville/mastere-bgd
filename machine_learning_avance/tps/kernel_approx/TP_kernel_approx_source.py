# Authors: Bellet, Gramfort, Salmon

from math import sqrt
import numpy as np

from scipy.sparse.linalg import svds
from scipy.linalg import svd

from sklearn.metrics.pairwise import rbf_kernel


def rank_trunc(gram_mat, k, fast=True):
    """
    k-th order approximation of the Gram Matrix G.

    Parameters
    ----------
    gram_mat : array, shape (n_samples, n_samples)
        the Gram matrix
    k : int
        the order approximation
    fast : bool
        use svd (if False) or svds (if True).

    Return
    ------
    gram_mat_k : array, shape (n_samples, n_samples)
        The rank k Gram matrix.
    """
    if fast:
        u, s, vt = svds(gram_mat, k)
        gram_mat_k = np.dot(np.dot(u, np.diag(s)), vt)
    else:
        u, s, vh = svd(gram_mat)
        gram_mat_k = np.dot(np.dot(u[:, :k], np.diag(s)[:k, :k]), vh[:k, :])

    return gram_mat_k


def random_features(X_train, X_test, gamma, c=300, seed=44):
    """Compute random kernel features

    Parameters
    ----------
    X_train : array, shape (n_samples1, n_features)
        The train samples.
    X_test : array, shape (n_samples2, n_features)
        The test samples.
    gamma : float
        The Gaussian kernel parameter
    c : int
        The number of components
    seed : int
        The seed for random number generation

    Return
    ------
    X_new_train : array, shape (n_samples1, c)
        The new train samples.
    X_new_test : array, shape (n_samples2, c)
        The new test samples.
    """
    rng = np.random.RandomState(seed)
    p = X_train.shape[1]
    W = rng.normal(0, 2*gamma, (p, c))
    b = rng.uniform(0, 2 * np.pi, c)

    f = np.sqrt(2 / c)
    X_new_train = f * np.cos(np.dot(X_train, W) + b)
    X_new_test = f * np.cos(np.dot(X_test, W) + b)
    # TODO Question 4
    return X_new_train, X_new_test


def nystrom(X_train, X_test, gamma, c=500, k=200, seed=44):
    """Compute nystrom kernel approximation

    Parameters
    ----------
    X_train : array, shape (n_samples1, n_features)
        The train samples.
    X_test : array, shape (n_samples2, n_features)
        The test samples.
    gamma : float
        The Gaussian kernel parameter
    c : int
        The number of points to sample for the approximation
    k : int
        The number of components
    seed : int
        The seed for random number generation

    Return
    ------
    X_new_train : array, shape (n_samples1, c)
        The new train samples.
    X_new_test : array, shape (n_samples2, c)
        The new test samples.
    """
    rng = np.random.RandomState(seed)
    n, p = X_train.shape
    I = rng.randint(0, n, c)
    X_train_I = X_train[I, :]
    g_tilde = rbf_kernel(X_train_I, X_train_I, gamma)
    u, l, _ = svd(g_tilde)
    u_k, l_k = u[:, :k], np.diag(l[:k])
    m_k = np.dot(u_k, np.sqrt(np.linalg.inv(l_k)))
    t_train = rbf_kernel(X_train, X_train_I)
    t_test = rbf_kernel(X_test, X_train_I)

    return np.dot(t_train, m_k), np.dot(t_test, m_k)
