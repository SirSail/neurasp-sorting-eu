import json
from pathlib import Path
import numpy as np

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def analyze_convergence(history, threshold=0.95):
    # Find epoch where acc_exact first crosses threshold
    for h in history:
        if h['acc_exact'] >= threshold:
            return h['epoch']
    return None

def analyze_size(results_dir, size):
    size_dir = results_dir / str(size)
    if not size_dir.exists():
        return None
        
    print(f"\n=== Analysis for Size {size} ===")
    
    print(f"\n=== Analysis for Size {size} ===")
    
    for model_dir in size_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        hist_path = model_dir / "history.json"
        if hist_path.exists():
            h = load_history(hist_path)
            if not h: continue
            
            final = h[-1]
            conv = analyze_convergence(h)
            
            name = model_dir.name
            if "neurasp" in name: name = name.replace("neurasp_", "NeurASP ")
            elif "neural" in name: name = name.replace("neural_", "Baseline ")
            
            print(f"{name}: Final Exact Acc: {final['acc_exact']:.4f}, F1: {final['f1']:.4f}, Time: {final['time']:.2f}s/epoch")
            print(f"          Convergence (>95%): {conv if conv else 'Not reached'}")

def main():
    results_dir = Path("results/experiments")
    
    # Real logic to find sizes
    available_sizes = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    def size_key(s):
        if s.endswith('k'): return int(s[:-1]) * 1000
        return int(s)
    
    available_sizes = sorted(available_sizes, key=size_key)
    
    for size in available_sizes:
        analyze_size(results_dir, size)

if __name__ == "__main__":
    main()
