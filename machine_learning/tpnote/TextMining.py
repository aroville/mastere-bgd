
# coding: utf-8

# In[1]:

import os.path as op
import numpy as np
from glob import glob
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk import word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool
from normalizr import Normalizr
from chronometer import Chronometer


# In[2]:

def read_f(f):
    return open(f).read()

with Chronometer() as chrono:
    # Load data
    uri = '/home/axel/mastere/machine_learning/tpnote/data'
    filenames_neg = sorted(glob(op.join(uri, 'imdb1', 'neg', '*.txt')))
    filenames_pos = sorted(glob(op.join(uri, 'imdb1', 'pos', '*.txt')))
    texts_neg = Pool().map(read_f, filenames_neg)
    texts_pos = Pool().map(read_f, filenames_pos)
    texts = texts_neg + texts_pos
print('Got texts in {:.3f}s'.format(float(chrono)))

y = np.ones(len(texts), dtype=np.int)
y[:len(texts_neg)] = 0.

print("%d documents" % len(texts))


# In[3]:

normalizr = Normalizr(language='en')
normalizr_options = [
    'remove_accent_marks',
    'replace_hyphens',
    'replace_punctuation',
    'replace_symbols',
    'remove_extra_whitespaces'
]


def normalize(t):
    return normalizr.normalize(t, normalizr_options).split(' ')

with Chronometer() as chrono:
    sw = open(uri + '/english.stop').read().splitlines()
print('Got stop words in {:.3f}s'.format(float(chrono)))


def not_in_sw(w):
    return w not in sw


def filter_sw(split):
    return list(filter(not_in_sw, split))


def split_text(text):
    return text.split(' ')

# In[4]:


def count_words(t, ignore_sw):
    with Chronometer() as chrono:
        splits = Pool().map(normalize, t)
        if ignore_sw:
            splits = Pool().map(filter_sw, splits)

        d = {w: i for i, w in enumerate(set(np.concatenate(splits)))}

        counts = np.zeros((len(t), len(d)))
        for ix_text, split in enumerate(splits):
            for word in split:
                counts[ix_text, d[word]] += 1
    print('Word count in {:.3f}s'.format(float(chrono)))

    return counts


# #2 Les classes positives et négatives ont été assignées à partir des notes données au film, avec une échelle différente selon le système de notation de la source.

# In[5]:

class NB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.prior = None
        self.condprobe = None

    def fit(self, x, y):
        '''
        Given x, a matrix of number of apparition by term and text,
        and y, a vector containing the labels associated with each text,
        calculate the frequency for each word, by label.
        '''
        with Chronometer() as chrono:
            n_docs, n_words = x.shape
            classes = np.unique(y)
            p = len(classes)

            # probability a priori
            self.prior = np.empty(p)
            self.condprobe = np.empty((p, n_words))

            for i, c in enumerate(classes):
                # over all the training data, frequency of label c
                self.prior[i] = len(y[y == c]) / n_docs

                # calculate the frequency of each word
                t = np.sum(x[y == c], axis=0)
                self.condprobe[i] = (t + 1) / np.sum(t + 1)

        print('Fit in {:.3f}s'.format(float(chrono)))

        return self

    def predict(self, x):
        '''
        Calculates a score for each label for a new "apparition matrix"
        Return the higher scoring labels
        '''
        with Chronometer() as chrono:
            proba = np.empty((x.shape[0], len(self.prior)))
            proba[:, :] = np.log(self.prior)

            # np.nonzeros allows to consider only non-zero terms
            self.condprobe = self.condprobe.T
            for c, t in np.transpose(np.nonzero(x)):
                proba[c] += np.log(self.condprobe[t])

        print('Predict in {:.3f}s'.format(float(chrono)))

        return np.argmax(proba, axis=1)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


# In[6]:

X_ignore_true = count_words(texts, ignore_sw=True)
X_ignore_false = count_words(texts, ignore_sw=False)


# In[7]:

def cross_val(naive_bayes, x):
    return np.mean(cross_val_score(naive_bayes, x, y, cv=5, n_jobs=-1))


def eval_nb(clf):
    nb = clf()
    print('\nScore using ' + clf.__name__)
    print('\tignore_sw=False =>', cross_val(nb, X_ignore_false))
    print('\tignore_sw=True =>', cross_val(nb, X_ignore_true))


# In[8]:

eval_nb(NB)


# In[9]:

eval_nb(MultinomialNB)


# In[8]:

vect = CountVectorizer(lowercase=True, stop_words={'english'})


def eval_pipeline_with_params(clf, va, ngr):
    pipeline = Pipeline([('vect', vect), ('clf', clf())])
    pipeline.set_params(vect__analyzer=va, vect__ngram_range=ngr)
    print(clf.__name__, va, ngr, cross_val(pipeline, texts))


def eval_pipeline(clf):
    params = [(clf, va, ngr)
              for va in ['word', 'char', 'char_wb']
              for ngr in [(1, 1), (1, 2)]]
    with Pool() as p:
        p.starmap(eval_pipeline_with_params, params)


# In[17]:

# eval_pipeline(MultinomialNB)


# In[18]:

# eval_pipeline(LinearSVC)


# In[19]:

# eval_pipeline(LogisticRegression)


# In[9]:

stemmer = SnowballStemmer('english', ignore_stopwords=False)
vect_stem = CountVectorizer(
    lowercase=True,
    stop_words=sw,
    tokenizer=lambda t: [stemmer.stem(token) for token in word_tokenize(t)],
    ngram_range=(1, 2),
    analyzer='word'
)


def eval_stem(clf):
    pipeline = Pipeline([('vect', vect_stem), ('clf', clf())])
    print(clf.__name__, ':', cross_val(pipeline, texts))


# In[21]:

# eval_stem(MultinomialNB)


# In[24]:

# eval_stem(LogisticRegression)


# In[10]:

# eval_stem(LinearSVC)


# In[8]:

ok_pos = [
    'JJ', 'JJR', 'JJS',                         # ADJECTIVES
    'NN', 'NNP', 'NNPS',                        # NOUNS
    'RB', 'RBR', 'RBS',                         # ADVERBS
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'     # VERBS
]


def filter_words(t):
    return [w for w, pos in pos_tag(word_tokenize(t)) if pos in ok_pos]

vect_pos = CountVectorizer(
    lowercase=True,
    stop_words=sw,
    tokenizer=filter_words,
    ngram_range=(1, 2),
    analyzer='word'
)


# In[9]:

# pipeline = Pipeline([('vect', vect_pos), ('clf', LogisticRegression())])
# print(LogisticRegression.__name__, ':', cross_val(pipeline, texts))


# In[ ]:


pipeline = Pipeline([
    ('vect', vect),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    ('clf', LinearSVC(tol=1e-3))])
print(LogisticRegression.__name__, ':', cross_val(pipeline, texts))
