import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

def load_history(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_learning_curves(results_dir):
    # Search for all history.json files
    # Structure: results/experiments/{size}/{model}/history.json
    
    sizes = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    # Sort helper: 100, 1k, 10k
    def size_key(s):
        if s.endswith('k'): return int(s[:-1]) * 1000
        return int(s)
    
    sizes = sorted(sizes, key=size_key)
    
    for size in sizes:
        size_dir = results_dir / size
        if not size_dir.exists(): continue
        
        plt.figure(figsize=(18, 5))
        
        # Subplot 1: Exact Accuracy
        plt.subplot(1, 3, 1)
        plt.title(f"Exact Match Accuracy (Size: {size})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        # Subplot 2: Element Accuracy
        plt.subplot(1, 3, 2)
        plt.title(f"Element Accuracy (Size: {size})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        # Subplot 3: F1 Score
        plt.subplot(1, 3, 3)
        plt.title(f"F1 Score (Size: {size})")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        
        for model_dir in size_dir.iterdir():
            if not model_dir.is_dir(): continue
            hist_path = model_dir / "history.json"
            if not hist_path.exists(): continue
            
            history = load_history(hist_path)
            epochs = [h['epoch'] for h in history]
            exact_acc = [h['acc_exact'] for h in history]
            elem_acc = [h.get('acc_element', 0) for h in history] # Safe get
            f1 = [h['f1'] for h in history]
            
            label = model_dir.name
            # Simplified label logic - rely on folder name which now contains hyperparams
            if "neurasp" in label:
                label = label.replace("neurasp_", "NeurASP ")
            elif "neural" in label:
                label = label.replace("neural_", "Baseline ")
                
            plt.subplot(1, 3, 1)
            plt.plot(epochs, exact_acc, label=label)
            
            plt.subplot(1, 3, 2)
            plt.plot(epochs, elem_acc, label=label)
            
            plt.subplot(1, 3, 3)
            plt.plot(epochs, f1, label=label)
            
        plt.subplot(1, 3, 1)
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.legend()
        
        out_path = results_dir / f"learning_curves_{size}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

def plot_time_comparison(results_dir):
    # Bar chart comparing Total Training Time across sizes
    sizes = []
    
    # Structure: { 'Baseline': [time_100, time_1k, ...], 'NeurASP_alpha0.1': [...] }
    # Since different sizes might have different models (though unlikely in standard runs), 
    # we need to be careful. For now, assuming consistent models across sizes for the bar chart 
    # might be tricky if config varies. 
    # Let's collect ALL model names found across all sizes.
    
    model_names = set()
    data_map = {} # { size: { model_name: time } }
    
    # Determine sizes
    raw_sizes = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    
    def size_key(s):
        if s.endswith('k'): return int(s[:-1]) * 1000
        return int(s)
        
    sorted_sizes = sorted(raw_sizes, key=size_key)
    
    for size in sorted_sizes:
        size_dir = results_dir / size
        data_map[size] = {}
        
        for model_dir in size_dir.iterdir():
            if not model_dir.is_dir(): continue
            hist_path = model_dir / "history.json"
            if hist_path.exists():
                h = load_history(hist_path)
                total_time = sum([e['time'] for e in h])
                data_map[size][model_dir.name] = total_time
                model_names.add(model_dir.name)

    # Sort model names for consistent plotting
    sorted_models = sorted(list(model_names))
    
    x = np.arange(len(sorted_sizes))
    width = 0.8 / len(sorted_models)
    
    plt.figure(figsize=(12, 6))
    
    for i, model_name in enumerate(sorted_models):
        times = []
        for size in sorted_sizes:
            times.append(data_map[size].get(model_name, 0)) # 0 if missing
            
        label = model_name
        if "neurasp" in label: label = label.replace("neurasp_", "NeurASP ")
        elif "neural" in label: label = label.replace("neural_", "Baseline ")
        
        plt.bar(x + (i * width) - (len(sorted_models)*width/2), times, width, label=label)
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Total Training Time (s)')
    plt.title('Total Training Time Comparison (Log Scale)')
    plt.xticks(x, sorted_sizes)
    plt.yscale('log') # Log scale because NeurASP is much slower
    plt.legend()
    
    out_path = results_dir / "time_comparison_bar_chart.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def plot_final_metrics_comparison(results_dir):
    # Bar chart comparing final Exact Match Acc across sizes
    
    model_names = set()
    data_map = {} # { size: { model_name: acc } }
    
    # Determine sizes
    raw_sizes = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    
    def size_key(s):
        if s.endswith('k'): return int(s[:-1]) * 1000
        return int(s)
        
    sorted_sizes = sorted(raw_sizes, key=size_key)
    
    for size in sorted_sizes:
        size_dir = results_dir / size
        data_map[size] = {}
        
        for model_dir in size_dir.iterdir():
            if not model_dir.is_dir(): continue
            hist_path = model_dir / "history.json"
            if hist_path.exists():
                h = load_history(hist_path)
                acc = h[-1]['acc_exact']
                data_map[size][model_dir.name] = acc
                model_names.add(model_dir.name)

    # Sort model names
    sorted_models = sorted(list(model_names))
    
    x = np.arange(len(sorted_sizes))
    width = 0.8 / len(sorted_models)
    
    plt.figure(figsize=(12, 6))
    
    for i, model_name in enumerate(sorted_models):
        accs = []
        for size in sorted_sizes:
            accs.append(data_map[size].get(model_name, 0))
            
        label = model_name
        if "neurasp" in label: label = label.replace("neurasp_", "NeurASP ")
        elif "neural" in label: label = label.replace("neural_", "Baseline ")
        
        plt.bar(x + (i * width) - (len(sorted_models)*width/2), accs, width, label=label)
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Final Exact Match Accuracy')
    plt.title('Baseline vs NeurASP Accuracy across Dataset Sizes')
    plt.xticks(x, sorted_sizes)
    plt.legend()
    
    out_path = results_dir / "comparison_bar_chart.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def main():
    results_dir = Path("results/experiments")
    if not results_dir.exists():
        print("No results found.")
        return
        
    plot_learning_curves(results_dir)
    plot_final_metrics_comparison(results_dir)
    plot_time_comparison(results_dir)

if __name__ == "__main__":
    main()
