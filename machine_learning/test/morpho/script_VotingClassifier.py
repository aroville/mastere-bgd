import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import pairwise_distances_argmin

X = pd.read_csv('data/training_templates.csv', header=None)
y = np.loadtxt('data/training_labels.txt', dtype=np.int)
X_pred = pd.read_csv('data/testing_templates.csv', header=None)

# Train on a set that is closer to the test set, to gain accuracy
distances = pairwise_distances_argmin(X_pred, X)
X = X.iloc[distances, :][:int(len(X) * 0.1)]
y = y[distances][:int(len(y) * 0.1)]

del distances

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


def compute_pred_score(y_true, y_pred):
    y_comp = y_true * y_pred
    return float(10 * np.sum(y_comp == -1) + np.sum(y_comp == 0)) / y_comp.shape[0]


def scoring(clf):
    pred_orig = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    best_score = 999.
    best_proba = 0.5
    for min_proba in np.arange(best_proba, 1., 0.01):
        pred = pred_orig
        indices = [i for i, p in enumerate(probas) if p[0] < min_proba and p[1] < min_proba]
        pred[indices] = 0
        score = compute_pred_score(y_test, pred)
        if score < best_score:
            best_proba = min_proba
            best_score = score

    return best_score, best_proba


def eval_clf(clf):
    clf.fit(X_train, y_train)

    sc, proba = scoring(clf)
    n_n = clf.estimators_[0].get_params()['n_neighbors']
    hls = clf.estimators_[1].get_params()['hidden_layer_sizes']
    svm = clf.estimators_[2].get_params()

    info = '_nn_' + str(n_n) + \
            '_hls_' + str(hls) + \
            '_svm_degree_' + str(svm['degree']) + \
            '_svm_kernel_' + str(svm['kernel']) + \
            '_proba_' + str(proba) + \
            '_sc_' + str(sc)

    print(info)

    probas = clf.predict_proba(X_pred)
    indices = [i for i, p in enumerate(probas) if p[0] < proba and p[1] < proba]

    y_pred = clf.fit(X, y).predict(X_pred)
    y_pred[indices] = 0
    np.savetxt('y_pred' + info + '.txt', y_pred, fmt='%d')
    return info


knn = KNeighborsClassifier(n_neighbors=3)
mlp = MLPClassifier(max_iter=1000, warm_start=True, hidden_layer_sizes=(200,))
svms = [SVC(degree=degree, kernel='poly', probability=True) for degree in [3, 4]]

clfs = [VotingClassifier(estimators=[
    ('knn', knn),
    ('mlp', mlp),
    ('svm', svm)], voting='soft') for svm in svms]

Pool().map(eval_clf, clfs)
