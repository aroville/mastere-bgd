import os.path as op
import numpy as np
from glob import glob
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from multiprocessing import Pool

# Load data
project_uri = '/home/axel/desktop/mastere/machine_learning/tpnote/data'
filenames_neg = sorted(glob(op.join(project_uri, 'imdb1', 'neg', '*.txt')))
filenames_pos = sorted(glob(op.join(project_uri, 'imdb1', 'pos', '*.txt')))

texts_neg = [open(f).read() for f in filenames_neg]
texts_pos = [open(f).read() for f in filenames_pos]
texts = texts_neg + texts_pos

y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))


def count_words(texts, ignore_stop_words=True):
    words = set(' '.join(texts).split(' '))

    if ignore_stop_words:
        stop_words = open(project_uri + '/english.stop').read()
        words = set(filter(lambda w: w not in stop_words, words))

    d = {w: i for i, w in enumerate(words)}

    counts = np.zeros((len(texts), len(words)))
    for ix_text, text in enumerate(texts):
        for word in text.split(' '):
            if not ignore_stop_words or word not in stop_words:
                counts[ix_text, d[word]] += 1

    return d, counts


class NB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.prior = None
        self.condprobe = None

    def fit(self, x, y):
        n_docs, n_words = x.shape
        classes = set(y)
        p = len(classes)
        self.prior = np.empty(p)
        self.condprobe = np.empty((p, n_words))

        for c in classes:
            self.prior[c] = len(y[y == c]) / n_docs
            t = np.sum(x[y == c], axis=0)
            self.condprobe[c] = (t + 1) / np.sum(t + 1)

        self.condprobe = self.condprobe.T
        return self

    def predict(self, x):
        score = np.empty((x.shape[0], len(self.prior)))
        score[:, :] = np.log(self.prior)

        for row, col in np.transpose(np.nonzero(x)):
            score[row] += np.log(self.condprobe[col])

        return np.argmax(score, axis=1)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


def eval_fold(sets):
    x_tr, y_tr, x_te, y_te = sets
    nb = NB().fit(x_tr, y_tr)
    score = nb.score(x_te, y_te)
    print(score)
    return score


def cross_val(t, y, n_splits, ignore_stop_words):
    _, x = count_words(t, ignore_stop_words=ignore_stop_words)
    kfolds = KFold(n_splits).split(x, y)
    sets = [[x[tr], y[tr], x[te], y[te]] for tr, te in kfolds]
    print('Mean:', np.mean(Pool().map(eval_fold, sets)))
    print('(ignoring stop words:', ignore_stop_words, ')\n\n')

cross_val(texts, y, 5, False)
cross_val(texts, y, 5, True)

