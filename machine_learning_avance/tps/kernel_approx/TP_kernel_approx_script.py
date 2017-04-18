# Authors: Bellet, Gramfort, Salmon

from time import time

from TP_kernel_approx_source import rank_trunc, nystrom, random_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC, LinearSVC

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###############################################################################
# Requires file ijcnn1.dat to be present in the directory

dataset_path = 'ijcnn1.dat'
ijcnn1 = load_svmlight_file(dataset_path)
X = ijcnn1[0].todense()
y = ijcnn1[1]

###############################################################################
# Extract features

X_train, X_test, y_train, y_test = train_test_split(
    X[:60000, :], y[:60000], train_size=20000, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###############################################################################
# SVM classification (Question 1)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

# LINEAR SVC (no kernel)
print("Fitting LinearSVC on %d samples..." % n_train)
t0 = time()
lsvc = LinearSVC(dual=False)
lsvc.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Predicting with LinearSVC on %d samples..." % n_test)
t1 = time()
y_pred_linear_svc = lsvc.predict(X_test)
print("done in %0.3fs" % (time() - t1))
timing_linear = time() - t0
accuracy_linear = np.mean(y_pred_linear_svc == y_test)
print("classification accuracy (LinearSVC): %0.3f" % accuracy_linear)
print()

# SVC (with kernel)
print("Fitting SVC rbf on %d samples..." % n_train)
t0 = time()
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Predicting with SVC rbf on %d samples..." % n_test)
t1 = time()
y_pred_svc = svc.predict(X_test)
print("done in %0.3fs" % (time() - t1))
timing_kernel = time() - t0
accuracy_kernel = np.mean(y_pred_svc == y_test)
print("classification accuracy (SVC): %0.3f" % accuracy_kernel)
print()

###############################################################################
# Gram approximation

p = 200
r_noise = 100
r_signal = 20

intensity = 50

rng = np.random.RandomState(42)
X_noise = rng.randn(r_noise, p)
X_signal = rng.randn(r_signal, p)

gram_signal = np.dot(X_noise.T, X_noise) + \
              intensity * np.dot(X_signal.T, X_signal)
n_ranks = 100
ranks = np.arange(1, n_ranks + 1)
rel_error = np.zeros(n_ranks)

# TODO : Question 2 Implement rank_trunc function in source file
if False:
    t1 = time()
    rank_trunc(gram_signal, 20, fast=True)
    print("Gram matrix with fast=True: %0.3fs" % (time() - t1))
    t1 = time()
    rank_trunc(gram_signal, 20, fast=False)
    print("Gram matrix with fast=False: %0.3fs" % (time() - t1))

# TODO : Question 3 Evaluate accuracy with Frobenius norm as a function
# of the rank for both svd solvers

# Use linalg.norm(A, 'fro') to compute Frobenius norm of A

if False:
    timing_fast, timing_slow = [], []
    rel_error_fast, rel_error_slow = [], []
    norm_g = linalg.norm(gram_signal, 'fro')

    for k in ranks:
        t = time()
        g_k_f = rank_trunc(gram_signal, k, fast=True)
        timing_fast.append(time() - t)
        rel_error_fast.append(linalg.norm(g_k_f - gram_signal) / norm_g)

        t = time()
        g_k_s = rank_trunc(gram_signal, k, fast=False)
        timing_slow.append(time() - t)
        rel_error_slow.append(linalg.norm(g_k_s - gram_signal) / norm_g)

    ###############################################################################
    # Display

    fig, axes = plt.subplots(ncols=1, nrows=2)
    ax1, ax2 = axes.ravel()

    ax1.plot(ranks, timing_fast, '-')
    ax1.plot(ranks, timing_slow, '-')

    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Time')
    ax2.plot(ranks, rel_error_fast, '-')
    ax2.plot(ranks, rel_error_slow, '-')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Relative Error')
    plt.tight_layout()
    plt.show()


###############################################################################
# Random Kernel Features:

n_samples, n_features = X_train.shape
n_samples_test, _ = X_test.shape
gamma = 1. / n_features

# TODO : Question 4 Implement random features in source file


# TODO : Question 5 Estimate training, testing time and accuracy
if False:
    Z_train, Z_test = random_features(X_train, X_test, gamma, c=300, seed=44)
    print("Fitting LinearSVC on %d samples..." % n_samples)
    t0 = time()
    clf = LinearSVC(dual=False)
    clf.fit(Z_train, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Predicting with SVC linear on %d samples..." % n_samples_test)
    t0 = time()
    accuracy = clf.score(Z_test, y_test)
    print("done in %0.3fs" % (time() - t0))
    print("classification accuracy: %0.3f" % accuracy)


###############################################################################
# SVM Nystrom:

# TODO : Question 6 Implement Nystrom in source file
if False:
    Z_train, Z_test = nystrom(X_train, X_test, gamma, c=500, k=200, seed=44)

    print("Fitting SVC linear on %d samples..." % n_samples)
    t0 = time()
    clf = LinearSVC(dual=False)
    clf.fit(Z_train, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Predicting with SVC linear on %d samples..." % n_samples_test)
    t0 = time()
    accuracy = clf.score(Z_test, y_test)
    print("done in %0.3fs" % (time() - t0))
    print("classification accuracy: %0.3f" % accuracy)
    print()


####################################################################
# Results / comparisons:

ranks = list(range(20, 750, 50))
n_ranks = len(ranks)
timing_rkf = np.zeros(n_ranks)
timing_nystrom = np.zeros(n_ranks)

accuracy_nystrom = np.zeros(n_ranks)
accuracy_rkf = np.zeros(n_ranks)

print("Training SVMs for various values of c...")
for i, c in enumerate(ranks):
    t0 = time()
    Z_train, Z_test = random_features(X_train, X_test, gamma, c=c, seed=44)
    clf = LinearSVC(dual=False)
    clf.fit(Z_train, y_train)
    accuracy_rkf[i] = clf.score(Z_test, y_test)
    timing_rkf[i] = time() - t0

    t0 = time()
    Z_train, Z_test = nystrom(X_train, X_test, gamma, c=c, k=c-10, seed=44)
    clf = LinearSVC(dual=False)
    clf.fit(Z_train, y_train)
    accuracy_nystrom[i] = clf.score(Z_test, y_test)
    timing_nystrom[i] = time() - t0

###############################################################################
# Display bis

fig, axes = plt.subplots(ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

ax1.plot(ranks, timing_nystrom, '-', label='Nystrom')
ax1.plot(ranks, timing_rkf, '-', label='RKF')
ax1.plot(ranks, timing_linear * np.ones(n_ranks), '-', label='LinearSVC')
ax1.plot(ranks, timing_kernel * np.ones(n_ranks), '-', label='RBF')

ax1.set_xlabel('Rank')
ax1.set_ylabel('Time')
ax1.legend(loc='lower right')

ax2.plot(ranks, accuracy_nystrom, '-', label='Nystrom')
ax2.plot(ranks, accuracy_rkf, '-', label='RKF')
ax2.plot(ranks, accuracy_linear * np.ones(n_ranks), '-', label='LinearSVC')
ax2.plot(ranks, accuracy_kernel * np.ones(n_ranks), '-', label='RBF')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='lower right')
plt.tight_layout()
plt.show()