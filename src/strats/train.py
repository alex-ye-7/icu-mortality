# Alexander Ye
# Training and eval for STraTS

import copy
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from triplet.strats import STraTS
from triplet.dataset import Dataset, DataLoader
from config import *
from shared.evaluate import evaluate_strats

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

def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--num_heads', type=int, default=N_HEAD)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--attention_dropout', type=float, default=DROPOUT)
    parser.add_argument('--V', type=int, default=len(STATIC_VARS) + len(TIME_SERIES_VARS))  # Vocab size
    parser.add_argument('--D', type=int, default=len(DEMO_FEATURES))  # Demographic embedding size (Age, Gender, Height, ICUType, Weight)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args = parse_args()

    # Split indices first (for normalization)
    y_train = torch.load(DATA_PROCESSED / 'y_train.pt', weights_only=True)
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.val_split))
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    # Load datasets (train stats computed from train_indices only)
    train_dataset = Dataset(split='train', stat_indices=train_indices)
    test_dataset = Dataset(split='test')
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, indices=train_indices)
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, indices=val_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = STraTS(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup training 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    # Training loop
    best_val_auroc = 0.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            probs = model(values=batch['values'], times=batch['times'],
                vars=batch['varis'], obs_mask=batch['obs_mask'], demo=batch['demo'])
            loss = criterion(probs, batch['labels'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_auroc, val_auprc = evaluate_strats(model, val_loader, device=device)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at Epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    # Restore best model before test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    roc_auc, pr_auc = evaluate_strats(model, test_loader, device=device)
    print(f"Test set AUROC: {roc_auc:.4f}")
    print(f"Test set AUPRC: {pr_auc:.4f}")
