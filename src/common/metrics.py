# Alexander Ye
# Common metrics for evaluation

import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix

def calc_auroc_auprc(y_true, y_pred):
    auroc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    return auroc, auprc

def calc_youdens(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificty = tn / (tn + fp)
    return sensitivity + specificty - 1

def find_optimal_youdens(y_true, y_pred): # y_pred is probabilities
    apply_threshold = lambda thres, preds: [1 if p > thres else 0 for p in preds]
    thresholds = np.linspace(0,1,101)
    scores = []
    for t in thresholds:
        y_temp = apply_threshold(t, y_pred)
        scores.append(calc_youdens(y_true, y_temp))
    best_t = thresholds[np.argmax(scores)]
    return best_t
