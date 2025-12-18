import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
from pathlib import Path
import sys
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.neural import SortingNetwork
from src.utils import calculate_metrics

def load_data(data_path):
    return torch.load(data_path, weights_only=True)

def train(args):
    device = torch.device("cpu") # explicit cpu as per requirements
    
    base_dir = Path(args.data_dir)
    
    # Load data
    train_X, train_Y = load_data(base_dir / "train.pt")
    val_X, val_Y = load_data(base_dir / "val.pt")
    
    # Dataset
    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = SortingNetwork(hidden_size=args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {args.data_dir} for {args.epochs} epochs.")
    
    history = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            out = model(X) # (B, 5, 5)
            
            # Reshape for CrossEntropy
            loss = criterion(out.view(-1, 5), Y.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Metrics
        metrics = calculate_metrics(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log entry
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": -1, # Cross entropy on val not explicitly calc'd in new utils, could add back if needed but metrics are key
            "acc_element": metrics["acc_element"],
            "acc_exact": metrics["acc_exact"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "time": epoch_time
        }
        history.append(log_entry)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Exact Acc: {metrics['acc_exact']:.4f} | "
                  f"F1: {metrics['f1']:.4f}")
                  
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")

    # Save Results
    results_dir = Path("results/experiments") / base_dir.name / f"neural_lr{args.lr}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model
    torch.save(model.state_dict(), results_dir / "model.pth")
    
    # Save History
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    # Save Summary
    summary = {
        "total_time": total_time,
        "final_metrics": history[-1],
        "config": vars(args)
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory (e.g. data/100)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=64)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
