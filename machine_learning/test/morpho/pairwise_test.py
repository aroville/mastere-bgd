import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

X = pd.read_csv('data/training_templates.csv', header=None)
y = np.loadtxt('data/training_labels.txt', dtype=np.int)
X_pred = pd.read_csv('data/testing_templates.csv', header=None)

distances_before = pairwise_distances_argmin_min(X, X_pred)
print('total distance:', np.sum(distances_before[1]))
X = X.ix[distances_before[0]][:int(len(X) * 0.2)]

distances_after = pairwise_distances_argmin_min(X, X_pred)
print('total distance:', np.sum(distances_after[1]))
