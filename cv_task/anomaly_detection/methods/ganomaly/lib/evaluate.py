""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
from copy import deepcopy
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def find_maximum(f, min_x, max_x, epsilon=1e-5):
    def binary_search(l, r, fl, fr, epsilon):
        mid = l + (r - l) / 2
        fm = f(mid)
        binary_search.eval_count += 1
        if (fm == fl and fm == fr) or r - l < epsilon:
            return mid, fm
        if fl > fm >= fr:
            return binary_search(l, mid, fl, fm, epsilon)
        if fl <= fm < fr:
            return binary_search(mid, r, fm, fr, epsilon)
        p1, f1 = binary_search(l, mid, fl, fm, epsilon)
        p2, f2 = binary_search(mid, r, fm, fr, epsilon)
        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    binary_search.eval_count = 0

    best_th, best_value = binary_search(min_x, max_x, f(min_x), f(max_x), epsilon)
    return best_th, best_value



def evaluate_all(labels, scores):
    roc_res = evaluate(labels, scores, 'roc')
    auprc_res = evaluate(labels, scores, 'auprc')
    f1_res = evaluate(labels, scores, 'f1_score')

    return roc_res, auprc_res, f1_res


def evaluate(labels, scores, metric='roc'):
    labels = deepcopy(labels)
    scores = deepcopy(scores)
    
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        min_t = scores.min()
        max_t = scores.max()
        
        def f(t):
            s = deepcopy(scores)
            s[s >= t] = 1
            s[s < t] = 0
            return f1_score(labels.cpu(), s.cpu())
        
        _, f1_res = find_maximum(f, min_t, max_t)
        return f1_res
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels.cpu(), scores.cpu())
    return ap
