import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown, plot_2d,
                              frontiere_new)

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)


n1 = 114
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data = rand_checkers(n1, n2, n3, n4, sigma)
X = np.array([[d[0], d[1]] for d in data])
y = np.array([int(d[2]) for d in data])
xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.8)

dmax = 30
dts_gini = [DTC(max_depth=d, criterion='gini').fit(xtr, ytr) for d in range(1, dmax+1)]
scores_gini = [dt.score(xte, yte) for dt in dts_gini]

dts_entropy = [DTC(max_depth=d, criterion='entropy').fit(xtr, ytr) for d in range(1, dmax+1)]
scores_entropy = [dt.score(xte, yte) for dt in dts_entropy]

# plt.figure(1)
# plt.plot(scores_gini)
# plt.plot(scores_entropy)
# plt.xlabel('Max depth')
# plt.ylabel('Accuracy Score')
# plt.show()

# dts = [dts_gini[-1]]
# for i, dt in enumerate(dts):  # _gini + dts_entropy):
#     # plt.subplot(6, 4, 1 + i)
#
#     params = dt.get_params()
#     criterion = params['criterion']
#     max_depth = params['max_depth']
#     plt.xlabel(criterion + ' ' + str(max_depth))
#
#     def f(xx):
#         """Classifier: needed to avoid warning due to shape issues"""
#         return dt.predict(xx.reshape(1, -1))
#     frontiere_new(f, X, y)
#     export_graphviz(dt, out_file='myTestTree1.dot', filled=True)
#     os.system("dot -Tpdf myTestTree1.dot -o myTestTree1.pdf")


# Q5 : Génération de base de test

n1 = 40
n2 = 40
n3 = 40
n4 = 40
data = rand_checkers(n1, n2, n3, n4, sigma)
X = np.array([[d[0], d[1]] for d in data])
y = np.array([int(d[2]) for d in data])
xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.8)

dmax = 30
dts_gini = [DTC(max_depth=d, criterion='gini').fit(xtr, ytr) for d in range(1, dmax+1)]
errors_gini = [np.sum(dt.predict(xte) != yte) for dt in dts_gini]

dts_entropy = [DTC(max_depth=d, criterion='entropy').fit(xtr, ytr) for d in range(1, dmax+1)]
errors_entropy = [np.sum(dt.predict(xte) != yte) for dt in dts_entropy]

print(errors_gini)
print(errors_entropy)

plt.figure(1)
plt.plot(range(1, dmax+1), errors_gini)
plt.plot(range(1, dmax+1), errors_entropy)
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.show()

# dts = [dts_gini[-1]]
# for i, dt in enumerate(dts):  # _gini + dts_entropy):
#     # plt.subplot(6, 4, 1 + i)
#
#     params = dt.get_params()
#     criterion = params['criterion']
#     max_depth = params['max_depth']
#     plt.xlabel(criterion + ' ' + str(max_depth))
#
#     def f(xx):
#         """Classifier: needed to avoid warning due to shape issues"""
#         return dt.predict(xx.reshape(1, -1))
#     # frontiere_new(f, X, y)
#     export_graphviz(dt, out_file='myTestTree1.dot', filled=True)
#     os.system("dot -Tpdf myTestTree1.dot -o myTestTree1.pdf")


# Q6. même question avec les données de reconnaissances de texte 'digits'
digits = datasets.load_digits()

n_samples = len(digits.data)
X = digits.data[:n_samples // 2]  # digits.images.reshape((n_samples, -1))
Y = digits.target[:n_samples // 2]
X_test = digits.data[n_samples // 2:]
Y_test = digits.target[n_samples // 2:]


#
# # Q7. estimer la meilleur profondeur avec un cross_val_score
#
# # TODO
#
# ############################################################################
# # Regression logistique
# ############################################################################
#
# # Q8. à Q12 appliquer la régression logistique aux digits
#
#
# symlist = ['o', 's', 'D', '+', 'x', '*', 'p', 'v', '-', '^']
# collist = ['blue', 'grey', 'red', 'purple', 'orange', 'salmon', 'black',
#            'fuchsia']
#
# logreg = linear_model.LogisticRegression()
#
# # TODO
