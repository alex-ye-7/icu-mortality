# Alexander Ye
# Training and eval for baseline models
# Usage: python -m src/baseline/train.py --model lstm --epochs 50

import copy
import random
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from baseline.utils import load_processed_data
from baseline.models import GRUPredictor, LSTMPredictor, TransformerPredictor
from common.metrics import calc_auroc_auprc
from config import *


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in dataloader: 
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred.append(model(batch_X).squeeze(-1).cpu().numpy())
            true.append(batch_y.cpu().numpy()) 
    
    true, pred = np.concatenate(true), np.concatenate(pred)
    auroc, auprc = calc_auroc_auprc(true, pred)
    return auroc, auprc, true, pred


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    best_val_auroc = 0.0
    epochs_without_improvement = 0
    best_state = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        val_auroc, val_auprc, _, _ = evaluate_model(model, val_loader, device)
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
    
    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'transformer'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args = parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_BASELINE)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[2]
    
    # Initialize model
    if args.model == 'lstm':
        model = LSTMPredictor(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT) 
    elif args.model == 'transformer':
        model = TransformerPredictor(input_size, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FF, DROPOUT)
    elif args.model == 'gru':
        model = GRUPredictor(input_size, HIDDEN_SIZE, NUM_LAYERS)
  
    model = model.to(device)
    
    # Data loader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    g = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_dataset, [1-args.val_split, args.val_split], generator=g)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
    
    # Test set performance!
    test_auroc, test_auprc, y_true, y_pred = evaluate_model(model, test_loader, device)
    print("="*50)
    print(f"Test Set: AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")

    # Save model
    # save_path = PROJECT_ROOT / "experiments" / f"{args.model}_epochs{args.epochs}.pt"
    # save_path.parent.mkdir(exist_ok=True)
    # torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")
