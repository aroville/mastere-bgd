import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # for plots
from sklearn import tree   # , datasets, metrics, cross_validation
import os
from matplotlib import rc


###############################################################################
# Plot initialization

rng = np.random.RandomState(42)
dirname = "../srcimages/"
imageformat = '.svg'
rc('font',
   **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 12,
          'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")

color_blind_list = sns.color_palette("colorblind", 8)
my_orange = color_blind_list[2]
my_green = color_blind_list[1]


###############################################################################
# Loss display






###############################################################################
# Toy data model: generated from a tree model!


def create_tree_data(n_samples, taux, tauy1, tauy2):
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=np.int8)
    for i in range(n_samples):
        x1, x2 = rng.rand(), rng.rand()
        print x1, x2
        X[i, :] = [x1, x2]
        if x1 < taux:
            if x2 < tauy1:
                y[i] = 1
            else:
                y[i] = 2
        else:
            if x2 < tauy2:
                y[i] = 2
            else:
                y[i] = 1
    return X, y

n_samples = 40
taux = 0.6
tauy1 = 0.8
tauy2 = 0.65

X, y = create_tree_data(n_samples, taux, tauy1, tauy2)


###############################################################################
# Train Decision Tree with sklearn:

my_classif = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
# Defautl choice for criterino in sklearn = 'gini'
my_classif.fit(X, y)


###############################################################################
# grid design with meshgrid for visualization of classification

def grid_2d(data, step=20):
    xmin, xmax = data[:, 0].min() - 1, data[:, 0].max() + 1
    ymin, ymax = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1. / step),
                         np.arange(ymin, ymax, (ymax - ymin) * 1. / step))
    return [xx, yy]

xx, yy = grid_2d(X, step=100)
Z = my_classif.predict(np.c_[xx.ravel(), yy.ravel()])
# rem: np.ravel returns a contiguous flattened array.
Z = Z.reshape(xx.shape)


###############################################################################
# Ploting and saving the dataset

plt.close("all")
fig1 = plt.figure(1, figsize=(5, 5))
ax = plt.gca()
plt.xlim(xmin=-0.1, xmax=1.1)
plt.ylim(ymin=-0.1, ymax=1.1)
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_xticklabels([])
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()

fig1.savefig('../srcimages/data_train_jo.svg')

###############################################################################
# Ploting and saving the Tree partition obtained

labs = np.unique(y)
idxbyclass = [np.where(y == labs[i])[0] for i in xrange(len(labs))]

fig2 = plt.figure(2, figsize=(5, 5))
ax = plt.gca()
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_xticklabels([])
symlist = ['D', 'o']
collist = [my_green, my_orange]

for i in range(len(labs)):
    plt.scatter(X[idxbyclass[i], 0], X[idxbyclass[i], 1],
                color=collist[i % len(collist)], cmap=plt.cm.Paired,
                marker=symlist[i % len(symlist)], s=200, edgecolors='k')

plt.xlim(xmin=-0.1, xmax=1.1)
plt.ylim(ymin=-0.1, ymax=1.1)
plt.show()

fig2.savefig('../srcimages/tree_train_jo.svg')


###############################################################################
# Ploting and saving both data and the Tree partition

fig2prime = plt.figure(1, figsize=(5, 5))
ax = plt.gca()
plt.xlim(xmin=-0.1, xmax=1.1)
plt.ylim(ymin=-0.1, ymax=1.1)
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_xticklabels([])
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for i in range(len(labs)):
    plt.scatter(X[idxbyclass[i], 0], X[idxbyclass[i], 1],
                color=collist[i % len(collist)], cmap=plt.cm.Paired,
                marker=symlist[i % len(symlist)], s=200, edgecolors='k')

plt.xlim(xmin=-0.1, xmax=1.1)
plt.ylim(ymin=-0.1, ymax=1.1)
plt.show()

fig2prime.savefig('../srcimages/tree_and_train_jo.svg')


###############################################################################
# Tree Visualization

tree.export_graphviz(my_classif, out_file="myTestTree.dot")
os.system("dot -Tpdf myTestTree.dot -o myTestTree.pdf")
os.system("evince myTestTree.pdf")


###############################################################################
###############################################################################
# Plot initialization for 1D curves

dirname = "../srcimages/"
imageformat = '.svg'
params = {'axes.labelsize': 12,
          'text.fontsize': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': (8, 6)}
plt.rcParams.update(params)
sns.set_context("poster")
sns.set_style("ticks")
sns.set_palette("colorblind")

rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)

###############################################################################
# Misclassication cost

x = np.arange(0.0001, 1.0001, step=0.0001)
misclaf_fun = lambda x: np.minimum(x, 1 - x)

fig3 = plt.figure(3, figsize=(6, 3))
ax1 = plt.gca()
ax1.plot([0, 0.5], [0.5, 0.5], 'k--', lw=2)
ax1.plot([0.5, 0.5], [0, 0.5], 'k--', lw=2)
ax1.plot(x, misclaf_fun(x))  # , label=r"Misclassification cost")
plt.xlim(xmin=0, xmax=1)
plt.ylim(ymin=0, ymax=0.5 * 1.05)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5])
plt.title('$\max =1/2$')
ax1.set_yticklabels(['', ''])
sns.despine()
plt.tight_layout()
plt.show()
###############################################################################
fig3.savefig('../srcimages/misclassification_cost.svg')


###############################################################################
# Entropy cost

x = np.arange(0.0001, 1.0001, step=0.0001)
entropie_fun = lambda x: - x * np.log(x) - (1 - x) * np.log(1 - x)

fig4 = plt.figure(4, figsize=(6, 3))
ax1 = plt.gca()
ax1.plot([0, 0.5], [np.log(2), np.log(2)], 'k--', lw=2)
ax1.plot([0.5, 0.5], [0, np.log(2)], 'k--', lw=2)
ax1.plot(x, entropie_fun(x))  # , label=r"Misclassification cost")
plt.xlim(xmin=0, xmax=1)
plt.ylim(ymin=0, ymax=np.log(2) * (1.05))
plt.xticks([0, 0.5, 1])
plt.yticks([0, np.log(2)])
plt.title('$\max = \log(2)$')
ax1.set_yticklabels(['', ''])
sns.despine()
plt.tight_layout()
plt.show()

fig4.savefig('../srcimages/entropy_cost.svg')


###############################################################################
# Gini cost

x = np.arange(0.0001, 1.0001, step=0.0001)
gini_fun = lambda x: 2 * x * (1 - x)

fig5 = plt.figure(5, figsize=(6, 3))
ax1 = plt.gca()
ax1.plot([0, 0.5], [0.5, 0.5], 'k--', lw=2)
ax1.plot([0.5, 0.5], [0, 0.5], 'k--', lw=2)
ax1.plot(x, gini_fun(x))  # , label=r"Misclassification cost")
plt.xlim(xmin=0, xmax=1)
plt.ylim(ymin=0, ymax=0.5 * (1.05))
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5])
plt.title('$\max = 1/8$')
ax1.set_yticklabels(['', ''])
sns.despine()
plt.tight_layout()
plt.show()

fig5.savefig('../srcimages/gini_cost.svg')
