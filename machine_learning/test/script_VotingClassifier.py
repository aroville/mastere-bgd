# coding: utf-8

import numpy as np
import pandas as pd
from multiprocessing import Manager
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Critere de performance
def compute_pred_score(y_true, y_pred):
    y_comp = y_true * y_pred
    score = float(10 * np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score

X = pd.read_csv('data/training_templates.csv', header=None)
y = np.loadtxt('data/training_labels.txt', dtype=np.int)
X_test = pd.read_csv('data/testing_templates.csv', header=None)

clf_proba = Manager().dict()


def scoring(clf, X, y):
    pred_orig = clf.predict(X)
    probas = clf.predict_proba(X)

    best_score = 999.
    best_proba = 0.7
    for min_proba in np.arange(best_proba, 1., 0.01):
        pred = pred_orig
        indices = [i for i, p in enumerate(probas) if p[0] < min_proba and p[1] < min_proba]
        pred[indices] = 0
        score = compute_pred_score(y, pred)
        if score < best_score:
            best_proba = min_proba
            best_score = score

    clf_proba[str(clf)] = [best_proba, best_score]
    print(clf, '\n\t', best_proba, '>>>', best_score)
    return -best_score


def eval_clf(clf):
    clf.fit(X, y)
    sc = np.max(cross_val_score(clf, X, y, scoring=scoring, cv=8, verbose=5, n_jobs=-1))
    return clf, sc


knns = [KNeighborsClassifier(n_neighbors=n_neighbors)
        for n_neighbors in [1, 3, 5, 7, 9, 11, 13]]
mlps = [MLPClassifier(max_iter=1000, warm_start=True, hidden_layer_sizes=(layers,))
        for layers in [20, 30, 50, 100, 200, 300]]

clfs = [VotingClassifier(estimators=[('knn', knn), ('mlp', mlp)], voting='soft')
        for knn in knns
        for mlp in mlps]

res = list(map(eval_clf, clfs))

max_score = -9999.
for clf_score in res:
    clf, score = clf_score
    if score > max_score:
        print(clf, '\n\tscore :', score)
        best_clf = clf
        max_score = score


print(best_clf)
joblib.dump(best_clf, 'best_clf.pkl')
best_clf.fit(X, y)

best_proba = clf_proba[str(best_clf)][0]
probas = [abs(n) for n in best_clf.predict_proba(X_test)]
indices = [i for i, p in enumerate(probas) if p[0] < best_proba and p[1] < best_proba]


y_pred = best_clf.predict(X_test)
y_pred[indices] = 0
# y_pred[indices] = 0
np.savetxt('y_pred.txt', y_pred, fmt='%d')
