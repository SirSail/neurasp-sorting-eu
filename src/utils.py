import torch
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(model, data_loader, device):
    """
    Evaluates the model on the given data_loader and returns a dictionary of metrics.
    Metrics:
    - loss (if applicable, implied by training loop, but here we focus on accuracy)
    - accuracy_element: Per-position accuracy
    - accuracy_exact: Exact match valid sort
    - precision, recall, f1: Macro averaged over classes (positions)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X) # (B, 5, 5) or (B, 25) depending on model, let's assume standard output
            
            # Check shape
            if output.dim() == 2 and output.size(1) == 25:
                 output = output.view(-1, 5, 5)
            
            # Predictions (indices)
            preds = torch.argmax(output, dim=2) # (B, 5)
            
            all_preds.append(preds.cpu())
            all_targets.append(Y.cpu())
            
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0) # (N, 5)
    all_targets = torch.cat(all_targets, dim=0) # (N, 5)
    
    # Elemental Accuracy
    correct_elements = (all_preds == all_targets).sum().item()
    total_elements = all_targets.numel()
    acc_element = correct_elements / total_elements
    
    # Exact Match Accuracy
    exact_matches = (all_preds == all_targets).all(dim=1).sum().item()
    total_samples = all_targets.size(0)
    acc_exact = exact_matches / total_samples
    
    # Precision, Recall, F1
    # Flatten for sklearn
    flat_preds = all_preds.view(-1).numpy()
    flat_targets = all_targets.view(-1).numpy()
    
    p, r, f1, _ = precision_recall_fscore_support(flat_targets, flat_preds, average='macro', zero_division=0)
    
    return {
        "acc_element": acc_element,
        "acc_exact": acc_exact,
        "precision": p,
        "recall": r,
        "f1": f1
    }
