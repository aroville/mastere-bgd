import os.path as op
import numpy as np
from glob import glob
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.help import upenn_tagset
from nltk import word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool

# Load data
uri = '/home/axel/desktop/mastere/machine_learning/tpnote/data'
filenames_neg = sorted(glob(op.join(uri, 'imdb1', 'neg', '*.txt')))
filenames_pos = sorted(glob(op.join(uri, 'imdb1', 'pos', '*.txt')))


def read_f(f):
    return open(f).read()

texts_neg = Pool().map(read_f, filenames_neg)
texts_pos = Pool().map(read_f, filenames_pos)
texts = texts_neg + texts_pos

y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))

sw = open(uri + '/english.stop').read().splitlines()


def not_in_sw(w):
    return w not in sw


def count_words(t, ignore_sw):
    all_words = set(' '.join(t).split(' '))
    dictionary = set(filter(not_in_sw, all_words)) if ignore_sw else all_words
    d = {w: i for i, w in enumerate(dictionary)}
    counts = np.zeros((len(t), len(d)))
    for ix_text, text in enumerate(t):
        split = text.split(' ')
        words = list(filter(not_in_sw, split)) if ignore_sw else split
        for word in words:
            counts[ix_text, d[word]] += 1

    return counts


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

        for i, c in enumerate(classes):
            self.prior[i] = len(y[y == c]) / n_docs
            t = np.sum(x[y == c], axis=0)
            self.condprobe[i] = (t + 1) / np.sum(t + 1)

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


def cross_val(naive_bayes, x, cv=5):
    return np.mean(cross_val_score(naive_bayes, x, y, cv=cv))


# Application
# X_ignore_true = count_words(texts, ignore_sw=True)
# X_ignore_false = count_words(texts, ignore_sw=False)

# for clf in [NB, MultinomialNB]:
#     nb = clf()
#     print('\nScore using ' + clf.__name__)
#     print('\tignore_sw=False =>', cross_val(nb, X_ignore_false))
#     print('\tignore_sw=True =>', cross_val(nb, X_ignore_true), '\n')

# vect = CountVectorizer(lowercase=True, stop_words=sw)
# for clf in [MultinomialNB, LinearSVC, LogisticRegression]:
#     print('Pipeline, MultinomialNB')
#     pipeline_nb = Pipeline([('vect', vect), ('clf', clf())])
#     for va in ['word', 'char', 'char_wb']:
#         for ngr in [(1, 1), (1, 2)]:
#             pipeline_nb.set_params(vect__analyzer=va, vect__ngram_range=ngr)
#             print(clf.__name__, va, ngr)
#             print(np.mean(cross_val(pipeline_nb, texts)), '\n')

# RESULT

# MultinomialNB
#               (1, 1)       (1, 2)
# word      |   0.8      |   0.8025
# char      |   0.6095   |   0.674
# char_wb   |   0.6115   |   0.6745

# LinearSVC
#               (1, 1)       (1, 2)
# word      |   0.8125   |   0.828
# char      |   0.5415   |   0.6835
# char_wb   |   0.5445   |   0.6055

# LogisticRegression
#               (1, 1)       (1, 2)
# word      |   0.829    |   0.836
# char      |   0.6375   |   0.7055
# char_wb   |   0.6385   |   0.7045


# (1, 2) et 'word' donnant les meilleurs résultats, nous nous limiterons
# à ces paramètres
stemmer = SnowballStemmer('english', ignore_stopwords=False)
# vect_stem = CountVectorizer(
#     lowercase=True,
#     stop_words=sw,
#     tokenizer=lambda t: [stemmer.stem(token) for token in word_tokenize(t)],
#     ngram_range=(1, 2),
#     analyzer='word'
# )
#
# for clf in [MultinomialNB, LogisticRegression, LinearSVC]:
#     pipeline = Pipeline([('vect', vect_stem), ('clf', clf())])
#     print(clf.__name__, ':', np.mean(cross_val(pipeline, texts)))

# RESULT
# MultinomialNB : 0.8115
# LogisticRegression : 0.838
# LinearSVC : 0.8315

# Que les noms, les verbes, les adverbes et les adjectifs
ok_pos = [
    'JJ', 'JJR', 'JJS',                         # ADJECTIVES
    'NN', 'NNP', 'NNPS',                        # NOUNS
    'RB', 'RBR', 'RBS',                         # ADVERBS
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'     # VERBS
]

# orig_stdout = sys.stdout
# sys.stdout = open('tags.txt', 'w+')
# upenn_tagset()
# sys.stdout = orig_stdout


def filter_words(t):
    return [w for w, pos in pos_tag(word_tokenize(t)) if pos in ok_pos]

vect_pos = CountVectorizer(
    lowercase=True,
    stop_words=sw,
    tokenizer=filter_words,
    ngram_range=(1, 2),
    analyzer='word'
)


pipeline = Pipeline([('vect', vect_pos), ('clf', LogisticRegression())])
print(LogisticRegression.__name__, ':', np.mean(cross_val(pipeline, texts)))

# RESULT
# LogisticRegression : 0.8355 (avec ou sans stem)
