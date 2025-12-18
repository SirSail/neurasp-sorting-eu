import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.neural import SortingNetwork

def load_data(data_path):
    return torch.load(data_path, weights_only=True)

def calculate_accuracy(outputs, targets):
    # outputs: (B, 5, 5) (logits)
    # targets: (B, 5) (indices)
    
    predictions = torch.argmax(outputs, dim=2) # (B, 5)
    
    # Per-element accuracy
    correct_elements = (predictions == targets).sum().item()
    total_elements = targets.numel()
    
    # Exact match (all 5 correct)
    exact_matches = (predictions == targets).all(dim=1).sum().item()
    total_samples = targets.size(0)
    
    return correct_elements / total_elements, exact_matches / total_samples

def evaluate(args):
    device = torch.device("cpu")
    
    base_dir = Path(args.data_dir)
    test_X, test_Y = load_data(base_dir / "test.pt")
    
    test_ds = TensorDataset(test_X, test_Y)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Load Model
    model = SortingNetwork(hidden_size=args.hidden_size).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()
    
    total_acc_elem = 0
    total_acc_exact = 0
    batches = 0
    
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            
            acc_elem, acc_exact = calculate_accuracy(out, Y)
            total_acc_elem += acc_elem
            total_acc_exact += acc_exact
            batches += 1
            
    avg_acc_elem = total_acc_elem / batches
    avg_acc_exact = total_acc_exact / batches
    
    print(f"Evaluation on {args.data_dir}")
    print(f"Element Accuracy: {avg_acc_elem:.4f}")
    print(f"Exact Match Accuracy: {avg_acc_exact:.4f}")
    
    results = {
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path),
        "element_accuracy": avg_acc_elem,
        "exact_accuracy": avg_acc_exact
    }
    
    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--out_file", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
