# Alexander Ye

"""
Usage: python src/evaluate.py --model_path experiments/lstm_epochs30.pt
"""

import argparse
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

from data_processing import load_processed_data
from models import LSTMPredictor, TransformerPredictor
from config import *

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_PROCESSED)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: model logic
    model = LSTMPredictor(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    
    # Evaluate
    train_auroc, train_auprc, _, _ = evaluate_model(model, X_train, y_train, device)
    test_auroc, test_auprc, y_pred, y_true = evaluate_model(model, X_test, y_test, device)
    
    print("="*50)
    print(f"Training Set: AUROC={train_auroc:.4f}, AUPRC={train_auprc:.4f}")
    print(f"Test Set: AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")

    # # Plot ROC and PR curves
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # # ROC Curve
    # fpr, tpr, _ = roc_curve(y_true, y_pred)
    # axes[0].plot(fpr, tpr, label=f'LSTM (AUROC = {test_auroc:.3f})')
    # axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    # axes[0].set_xlabel('False Positive Rate')
    # axes[0].set_ylabel('True Positive Rate')
    # axes[0].set_title('ROC Curve')
    # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)

    # # Precision-Recall Curve
    # precision, recall, _ = precision_recall_curve(y_true, y_pred)
    # axes[1].plot(recall, precision, label=f'LSTM (AUPRC = {test_auprc:.3f})')
    # axes[1].axhline(y=y_true.mean(), color='k', linestyle='--',
    #                 label=f'Baseline ({y_true.mean():.3f})')
    # axes[1].set_xlabel('Recall')
    # axes[1].set_ylabel('Precision')
    # axes[1].set_title('Precision-Recall Curve')
    # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()