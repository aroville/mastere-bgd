from itertools import chain
import nltk
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from collections import Counter

nltk.corpus.conll2002.fileids()

train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        # 'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2]
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word2.lower(),
            '-2:word.istitle=%s' % word2.istitle(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:postag=' + postag2,
            '-2:postag[:2]=' + postag2[:2],
            ])

    if i > 2:
        word3 = sent[i-2][0]
        postag3 = sent[i-2][1]
        features.extend([
            '-3:word.lower=' + word3.lower(),
            '-3:word.istitle=%s' % word3.istitle(),
            '-3:word.isupper=%s' % word3.isupper(),
            '-3:postag=' + postag3,
            '-3:postag[:2]=' + postag3[:2],
            ])

    if i < len(sent)-3:
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        features.extend([
            '+3:word.lower=' + word3.lower(),
            '+3:word.istitle=%s' % word3.istitle(),
            '+3:word.isupper=%s' % word3.isupper(),
            '+3:postag=' + postag3,
            '+3:postag[:2]=' + postag3[:2],
            ])
        
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:postag=' + postag2,
            '+2:postag[:2]=' + postag2[:2],
        ])

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset)

c1 = 0.001
c2 = 0.001
trainer.set_params({
    'c1': c1,   # coefficient for L1 penalty
    'c2': c2,   # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
trainer.train('conll2002-esp.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('conll2002-esp.crfsuite')

y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))
    

trainer.train('conll2002-esp.crfsuite')

# tagger = pycrfsuite.Tagger()
# tagger.open('conll2002-esp.crfsuite')
example_sent = test_sents[0]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))

# y_pred = [tagger.tag(xseq) for xseq in X_test]


# ..and check the result. Note this report is not comparable to results in
# CONLL2002 papers because here we check per-token results (not per-entity).
# Per-entity numbers will be worse.
# print(bio_classification_report(y_test, y_pred))


# ## Let's check what classifier learned
info = tagger.info()


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

# print("Top likely transitions:")
# print_transitions(Counter(info.transitions).most_common(15))
#
# print("\nTop unlikely transitions:")
# print_transitions(Counter(info.transitions).most_common()[-15:])


# Check the state features:
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))
# print_state_features(Counter(info.state_features).most_common())

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])
