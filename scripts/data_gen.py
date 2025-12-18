import torch
import numpy as np
import argparse
from pathlib import Path

def generate_batch(size: int, n_numbers: int = 5, min_val: int = 0, max_val: int = 100):
    """
    Generates a batch of data.
    X: (size, n_numbers) - Random integers
    Y: (size, n_numbers) - Ranks (position in sorted array)
    """
    X_list = []
    Y_list = []
    
    for _ in range(size):
        # Generate 5 unique integers
        x = np.random.choice(range(min_val, max_val), size=n_numbers, replace=False)
        # Ranks: argsort(argsort(x))
        y = np.argsort(np.argsort(x))
        
        X_list.append(x)
        Y_list.append(y)
        
    return torch.tensor(np.array(X_list), dtype=torch.long), torch.tensor(np.array(Y_list), dtype=torch.long)

def save_dataset(base_dir: Path, size: int):
    print(f"Generating dataset with size {size}...")
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    # don't need test_ratio, because it get's calculated below 

    n_train = int(size * train_ratio)
    n_val = int(size * val_ratio)
    n_test = size - n_train - n_val

    # Generate full dataset
    X, Y = generate_batch(size)
    
    # Shuffle
    perm = torch.randperm(size)
    X = X[perm]
    Y = Y[perm]

    # Split
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    # Save
    size_str = str(size)
    if size >= 1000:
        size_str = f"{size//1000}k"
    
    save_path = base_dir / size_str
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save((X_train, Y_train), save_path / "train.pt")
    torch.save((X_val, Y_val), save_path / "val.pt")
    torch.save((X_test, Y_test), save_path / "test.pt")
    
    print(f"Saved {size_str} dataset to {save_path}")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 10000], help="Dataset sizes to generate")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    base_dir = Path(args.out_dir)
    
    for size in args.sizes:
        save_dataset(base_dir, size)

if __name__ == "__main__":
    main()
