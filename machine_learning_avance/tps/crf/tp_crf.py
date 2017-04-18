# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 14:38:23 2015

@author: essid
"""
import pickle as pickle
import numpy as np
from pycrfsuite import Tagger
from flexcrf_tp.crfsuite2flexcrf import convert_data_to_flexcrf
import warnings
from flexcrf_tp.models.linear_chain import (_feat_fun_values,
                                            _compute_all_potentials,
                                            _forward_score,
                                            _backward_score,
                                            _partition_fun_value,
                                            _posterior_score)

warnings.filterwarnings("ignore")


def viterbi_decoder(m_xy, n=None, log_version=True):
    """
    Performs MAP inference, determining $y = \argmax_y P(y|x)$, using the
    Viterbi algorithm.

    Parameters
    ----------
    m_xy : ndarray, shape (n_obs, n_labels, n_labels)
        Values of log-potentials ($\log M_i(y_{i-1}, y_i, x)$)
        computed based on feature functions f_xy and/or user-defined potentials
        `psi_xy`. At t=0, m_xy[0, 0, :] contains values of $\log M_1(y_0, y_1)$
        with $y_0$ the fixed initial state.

    n : integer, default=None
        Time position up to which to decode the optimal sequence; if not
        specified (default) the score is computed for the whole sequence.

    Returns
    -------
    y_pred : ndarray, shape (n_obs,)
        Predicted optimal sequence of labels.

    """
    # Initialization
    n_obs, n_labels, _ = m_xy.shape
    delta = np.zeros((n_obs, n_labels))
    psi = np.zeros((n_obs, n_labels))

    delta[0, :] = m_xy[0, 0, :]

    # Recursion
    for m in range(1, n_obs):
        for s in range(n_labels):
            current_exploration = delta[m-1, :] + m_xy[m, :, s]
            psi[m, s] = np.argmax(current_exploration)
            delta[m, s] = current_exploration[psi[m, s]]

    # Termination
    y_pred = np.zeros(n_obs)
    y_pred[-1] = np.argmax(delta[-1, :])

    # Path backtracking
    for m in range(n_obs-2, -1, -1):
        y_pred[m] = psi[m+1, y_pred[m+1]]

    return y_pred.astype(np.int64)

# -- Load data and crfsuite model and convert them-------------------------

RECREATE = True  # set to True to recreate flexcrf data with new model


CRFSUITE_MODEL_FILE = 'conll2002-esp.crfsuite'
CRFSUITE_TEST_DATA_FILE = 'conll2002-esp_crfsuite-test-data.dump'
FLEXCRF_TEST_DATA_FILE = 'conll2002-esp_flexcrf-test-data.dump'

# crfsuite model
tagger = Tagger()
tagger.open(CRFSUITE_MODEL_FILE)
model = tagger.info()

data = pickle.load(open(CRFSUITE_TEST_DATA_FILE, 'rb'))
print("test data loaded.")

if RECREATE:
    dataset, thetas = convert_data_to_flexcrf(data, model, n_seq=3)
    pickle.dump({'dataset': dataset, 'thetas': thetas},
                open(FLEXCRF_TEST_DATA_FILE, 'wb'))
else:
    dd = pickle.load(open(FLEXCRF_TEST_DATA_FILE))
    dataset = dd['dataset']
    thetas = dd['thetas']

# -- Start classification ------------------------------------------------

for seq in range(len(dataset)):
    # -- with crfsuite
    s_ = tagger.tag(data['X'][seq])
    y_ = np.array([int(model.labels[s]) for s in s_])
    # prob_ = tagger.probability(s_)

    # print("\n-- With CRFSUITE:")
    # print("labels:\n", s_, "\n", y_)
    # print("probability:\t %f" % prob_)

    # -- with flexcrf
    f_xy, y = dataset[seq]
    theta = thetas[seq]

    m_xy, f_m_xy = _compute_all_potentials(f_xy, theta)

    y_pred = viterbi_decoder(m_xy)

    alpha = _forward_score(m_xy=m_xy)
    beta = _backward_score(m_xy=m_xy)
    z_x = _partition_fun_value(alpha=alpha)
    f_x = _feat_fun_values(f_xy=f_xy, y=y_pred, with_f_x_sum=False)

    prob = _posterior_score(f_x=f_x, theta=theta, z_x=z_x, log_version=False)

    print("-- With FLEXCRF:")
    # print("labels:\n", y_pred)
    print("probability:\t %f" % prob)
    # print()
    print("equal predictions: ", np.all(y_pred == y_))
    # print("delta:\t %f" % abs(prob-prob_))

tagger.close()
