# coding: utf-8

import numpy as np
import pandas as pd
from multiprocessing import Manager
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Critere de performance
def compute_pred_score(y_true, y_pred):
    y_comp = y_true * y_pred
    score = float(10 * np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score

X = pd.read_csv('data/training_templates.csv', header=None)
y = np.loadtxt('data/training_labels.txt', dtype=np.int)
X_test = pd.read_csv('data/testing_templates.csv', header=None)

clf_percentiles = Manager().dict()


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

    clf_percentiles[str(clf)] = [best_proba, best_score]
    print(clf, '\n\t', best_proba, '>>>', best_score)
    return -best_score


def eval_clf(clf):
    sc = np.max(cross_val_score(clf, X, y, scoring=scoring, cv=8, verbose=5, n_jobs=-1))
    return clf, sc

clfs = [AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators=n_estimators)
        for base_estimator in [DecisionTreeClassifier()]
        for n_estimators in [50, 100, 200]
    ]

knns = list(map(eval_clf, clfs))

max_score = -9999.
for knn_score in knns:
    knn, score = knn_score
    if score > max_score:
        print(knn, '\n\tscore :', score)
        best_clf = knn
        max_score = score


print(best_clf)  # , '>>>>>>', max_score)
joblib.dump(best_clf, 'best_clf.pkl')
best_clf.fit(X, y)

best_percentile = clf_percentiles[str(best_clf)][0]
distances = [abs(n) for n in best_clf.decision_function(X_test)]
indices = [i for i, p in enumerate(distances) if p < best_percentile]


y_pred = best_clf.predict(X_test)
y_pred[indices] = 0
# y_pred[indices] = 0
np.savetxt('y_pred.txt', y_pred, fmt='%d')
