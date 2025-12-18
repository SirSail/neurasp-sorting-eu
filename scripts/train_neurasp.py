import torch
import argparse
from pathlib import Path
import sys
import time
import json
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.neurasp_model import SymbolicSorter
from src.utils import calculate_metrics

def load_data(data_path):
    return torch.load(data_path, weights_only=True)

def prepare_neurasp_data(X, Y):
    """
    Convert (Size, 5) tensors to NeurASP data format.
    dataList: [{'vals': (X[i], {'neural_pos': Y[i]})}, ...]
    obsList: ["val(0, X[i][0]). ...", ...]
    """
    data_list = []
    obs_list = []
    
    for i in range(len(X)):
        vals = X[i]
        ranks = Y[i]
        
        # Data dictionary item
        # t='vals' maps to (tensor(1,5) or (5), dictionary of labels)
        data_item = {
            'vals': (vals.unsqueeze(0), {'neural_pos': ranks})
        }
        data_list.append(data_item)
        
        # Observation (Facts about values)
        # val(Index, Value).
        obs = ""
        for idx, v in enumerate(vals):
            obs += f"val({idx}, {v.item()}). "
        obs_list.append(obs)
        
    return data_list, obs_list

def train(args):
    device = torch.device("cpu")
    base_dir = Path(args.data_dir)
    
    print("Loading data...")
    train_X, train_Y = load_data(base_dir / "train.pt")
    val_X, val_Y = load_data(base_dir / "val.pt")
    
    # Prepare Validation Loader for standard PyTorch evaluation
    val_ds = TensorDataset(val_X, val_Y)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Prepare Training Data for NeurASP
    print("Preparing data for NeurASP...")
    data_list, obs_list = prepare_neurasp_data(train_X, train_Y)
    
    print("Initializing NeurASP model...")
    model = SymbolicSorter(device=device, lr=args.lr)
    
    print(f"Starting training for {args.epochs} epochs with alpha={args.alpha}...")
    
    history = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # NeurASP Step (1 epoch)
        # Note: NeurASP class doesn't easily expose loss for return, but we can verify learning via metrics
        model.learn(data_list, obs_list, epochs=1, alpha=args.alpha, batch_size=args.batch_size)
    
        # Evaluation
        # We access the underlying neural model: model.model
        metrics = calculate_metrics(model.model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        log_entry = {
            "epoch": epoch + 1,
            "acc_element": metrics["acc_element"],
            "acc_exact": metrics["acc_exact"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "time": epoch_time
        }
        history.append(log_entry)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Exact Acc: {metrics['acc_exact']:.4f} | "
                  f"F1: {metrics['f1']:.4f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")
    
    # Save Results
    results_dir = Path("results/experiments") / base_dir.name / f"neurasp_alpha{args.alpha}_lr{args.lr}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model
    model.save(results_dir / "model.pth")
    
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
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of CrossEntropy loss. 1-alpha is logic loss.")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
