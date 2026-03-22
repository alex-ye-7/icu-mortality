# Alexander Ye

"""
Usage: python src/train.py --model lstm --epochs 50
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from utils.base_utils import load_processed_data
from baselines.base_models import *
from config import *
from evaluate import evaluate_model
from sklearn.metrics import confusion_matrix

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'transformer'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_PROCESSED)
    
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    model = train_model(model, train_loader, criterion, optimizer, device, args.epochs)
    
    # train_auroc, train_auprc, _, _ = evaluate_model(model, X_train, y_train, device)
    # test_auroc, test_auprc, y_pred, y_true = evaluate_model(model, X_test, y_test, device)
    
    # print("="*50)
    # print(f"Training Set: AUROC={train_auroc:.4f}, AUPRC={train_auprc:.4f}")
    # print(f"Test Set: AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f}")

    # Save model
    save_path = PROJECT_ROOT / "experiments" / f"{args.model}_epochs{args.epochs}.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()