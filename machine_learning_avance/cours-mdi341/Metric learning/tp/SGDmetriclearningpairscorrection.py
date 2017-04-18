import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import metrics
import itertools
import sys

plt.close('all')

############################################################################
#            Loading and visualizing the data
############################################################################

if 1:  # use iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
else:  # use digits
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # on ne garde que les 5 premieres classes par simplicite
    X = X[y < 5]
    y = y[y < 5]

# standardize data
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[np.isnan(X)] = 0.


def plot_2d(X, y):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""
    plt.figure()
    symlist = ['o', 's', '*', 'x', 'D', '+', 'p', 'v', 'H', '^']
    collist = ['blue', 'red', 'purple', 'orange', 'salmon', 'black', 'grey',
               'fuchsia']

    labs = np.unique(y)
    idxbyclass = [y == labs[i] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.plot(X[idxbyclass[i], 0], X[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(X[:, 1]), np.max(X[:, 1])])
    plt.xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    plt.show()

############################################################################
#            Displaying labeled data
############################################################################

# on utilise PCA pour projeter les donnees en 2D
pca = decomposition.PCA(n_components=2)
X_2D = pca.fit_transform(X)
plot_2d(X_2D, y)

############################################################################
#                Stochastic gradient for metric learning
############################################################################


def psd_proj(M):
    """ projection de la matrice M sur le cone des matrices semi-definies
    positives"""
    # calcule des valeurs et vecteurs propres
    eigenval, eigenvec = np.linalg.eigh(M)
    # on trouve les valeurs propres negatives ou tres proches de 0
    ind_pos = eigenval > 1e-10
    # on reconstruit la matrice en ignorant ces dernieres
    M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
               eigenvec[:, ind_pos].T)
    return M


def hinge_loss_pairs(X, pairs_idx, y_pairs, M):
    """Calcul du hinge loss sur les paires
    """
    diff = X[pairs_idx[:, 0], :] - X[pairs_idx[:, 1], :]
    return np.maximum(0., 1. + y_pairs.T * (np.sum(np.dot(M, diff.T) * diff.T,
                                                   axis=0) - 2.))


def sgd_metric_learning_pairs(X, y, gamma, alpha, n_iter, n_eval, M_ini,
                              random_state=42):
    """Stochastic gradient algorithm for metric learning with pairs

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets.
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    alpha : float
        The regularization parameter
    n_iter : int
        The number of iterations
    n_eval : int
        The number of pairs to evaluate the objective function
    M_ini : array, shape (n_features,n_features)
        The initial value of M
    random_state : int
        Random seed to make the algorithm deterministic
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    # tirer n_eval paires aleatoirement
    pairs_idx = rng.randint(0, n_samples, (n_eval, 2))
    # calcul du label des paires
    y_pairs = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
    M = M_ini.copy()
    pobj = np.zeros(n_iter)

    if not callable(gamma):
        def gamma_func(t):
            return gamma
    else:
        gamma_func = gamma

    for t in range(n_iter):
        pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx,
                                           y_pairs, M)) + alpha * np.trace(M)
        idx = rng.randint(0, n_samples, 2)
        diff = X[idx[0], :] - X[idx[1], :]
        y_idx = 2.0 * (y[idx[0]] == y[idx[1]]) - 1.0
        gradient = (y_idx * np.outer(diff, diff) *
                    ((1. + y_idx * (np.dot(diff, M.dot(diff.T)) - 2.)) > 0))
        gradient += alpha * np.eye(n_features)
        M -= gamma_func(t) * gradient
        M = psd_proj(M)
    return M, pobj


def combs(a, r):
    """ compute all r-length combinations of elements in array; a faster
    than np.array(list(itertools.combinations(a, r)))
    """
    a = np.asarray(a)
    dt = np.dtype([('', a.dtype)] * r)
    b = np.fromiter(itertools.combinations(a, r), dt)
    return b.view(a.dtype).reshape(-1, r)


def sgd_metric_learning_pairs_minibatch(X, y, gamma, alpha, n_iter, n_eval,
                                        n_batch, batch_type, M_ini,
                                        random_state=42):
    """Mini-batch stochastic gradient algorithm for metric learning with pairs

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets.
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    alpha : float
        The regularization parameter
    n_iter : int
        The number of iterations
    n_eval : int
        The number of pairs to evaluate the objective function
    n_batch : int
        Size of *subsample* for mini-batch (n_batch_pairs is given by
        n_batch*(n_batch-1)/2)
    batch_type : int
        0 if sample pairs directly, 1 if build pairs on subsample
    M_ini : array, shape (n_features,n_features)
        The initial value of M
    random_state : int
        Random seed to make the algorithm deterministic
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    # tirer n_eval paires aleatoirement
    pairs_idx = rng.randint(0, n_samples, (n_eval, 2))
    # calcul du label des paires
    y_pairs = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
    M = M_ini.copy()
    pobj = np.zeros(n_iter)

    # number of pairs in the mini-batch
    n_batch_pairs = n_batch * (n_batch - 1) / 2

    if not callable(gamma):
        def gamma_func(t):
            return gamma
    else:
        gamma_func = gamma

    for t in range(n_iter):
        pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx,
                                           y_pairs, M)) + alpha * np.trace(M)
        if batch_type == 0:
            idx = rng.randint(0, n_samples, (n_batch_pairs, 2))
        elif batch_type == 1:
            idx_sample = rng.randint(0, n_samples, n_batch)
            idx = combs(idx_sample, 2)
        else:
            sys.exit('Error: batch_type parameter should be 0 or 1')
        diff = X[idx[:, 0], :] - X[idx[:, 1], :]
        y_idx = 2.0 * (y[idx[:, 0]] == y[idx[:, 1]]) - 1.0
        gradient = np.zeros((n_features, n_features))
        slack = (1. + y_idx.T * (np.sum(np.dot(M, diff.T) * diff.T,
                                 axis=0) - 2.)) > 0
        for i in range(0, n_batch_pairs):
            if slack[i]:
                gradient += y_idx[i] * np.outer(diff[i, :], diff[i, :])
        gradient /= n_batch_pairs
        gradient += alpha * np.eye(n_features)
        M -= gamma_func(t) * gradient
        M = psd_proj(M)
    return M, pobj


n_features = X.shape[1]

M_ini = np.eye(n_features)
# M, pobj = sgd_metric_learning_pairs(X, y, 0.002, 0.0, 10000, 1000, M_ini)
M, pobj = sgd_metric_learning_pairs_minibatch(X, y, 0.002, 0.0, 10000, 1000,
                                              5, 0, M_ini)

plt.figure()
plt.plot(pobj)
plt.xlabel('t')
plt.ylabel('cost')
plt.title('hinge stochastic with pairs')
plt.show()

# check number of nonzero eigenvalues
e, v = np.linalg.eig(M)
print "Nb de valeurs propres non nulles de M: ", np.sum(e > 1e-12),\
      "/", e.shape[0]


# calcul de la factorisation de cholesky
# on ajoute de tres faibles coefficients sur la diagonale pour eviter
# les erreurs numeriques
L = np.linalg.cholesky(M + 1e-10 * np.eye(n_features))
# on projette lineairement les donnees
X_proj = np.dot(X, L)

# on utilise PCA pour projeter les donnees en 2D
X_proj_2D = pca.fit_transform(X_proj)

plot_2d(X_proj_2D, y)


# tirer des paires aleatoires
pairs = np.random.randint(0, X.shape[0], (10000, 2))
y_pairs = 2.0 * (y[pairs[:, 0]] == y[pairs[:, 1]]) - 1.0
diff = X[pairs[:, 0], :] - X[pairs[:, 1], :]
dist_euc = np.sqrt(np.sum(diff ** 2, axis=1))
dist_M = np.sum(np.dot(M, diff.T) * diff.T, axis=0)

# compute ROC curve
fpr_euc, tpr_euc, thresh_euc = metrics.roc_curve(y_pairs, -dist_euc)

# compute AUC
auc_euc = metrics.auc(fpr_euc, tpr_euc)

# Now with learnt metric
fpr_M, tpr_M, thresh_M = metrics.roc_curve(y_pairs, -dist_M)

# compute AUC
auc_M = metrics.auc(fpr_M, tpr_M)

# plot ROC curves
plt.figure()
plt.plot(fpr_euc, tpr_euc, label='Euclidean distance - AUC %.2f' % auc_euc)
plt.plot(fpr_M, tpr_M, label='Learnt distance - AUC %.2f' % auc_M)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
