# Alexander Ye
# Functions for evaluation

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, confusion_matrix

def evaluate_model(model, X, y, device):
    """Evaluate model and return metrics."""
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y_pred = model(X).squeeze().cpu().numpy()
        y_true = y.numpy()
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return auroc, auprc, y_pred, y_true

def evaluate_strats(model, dataloader, device):
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for batch_test in dataloader:
            batch_test = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_test.items()}
            probs = model(values=batch_test['values'], times=batch_test['times'],
                vars=batch_test['varis'], obs_mask=batch_test['obs_mask'],
                demo=batch_test['demo']
            )
            true.append(batch_test['labels'])
            pred.append(probs)
            
        true, pred = torch.cat(true).cpu().numpy(), torch.cat(pred).cpu().numpy()
        roc_auc = roc_auc_score(true, pred)
        precision, recall, _ = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        return roc_auc, pr_auc

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
