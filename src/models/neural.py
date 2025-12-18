import torch
import torch.nn as nn

class SortingNetwork(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_numbers=5):
        super(SortingNetwork, self).__init__()
        self.num_numbers = num_numbers
        
        # Simple MLP
        # Input: 5 integers (values)
        # Output: 5 * 5 logits (for each number, a distribution over 5 positions)
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_numbers * num_numbers)
        )
        self.output_probs = False

    def set_output_probs(self, mode=True):
        self.output_probs = mode

        
    def forward(self, x):
        # x: (B, 5) values
        # We might need to normalize input if values are large, but for 0-100 it's likely okay-ish 
        # or we could add a BatchID layer or just use float.
        
        x = x.float()
        flat_out = self.net(x) # (B, 25)
        
        # Reshape to (B, 5, 5)
        # Dim 1: The 5 input numbers
        # Dim 2: The 5 possible positions
        out = flat_out.view(-1, self.num_numbers, self.num_numbers)
        
        if self.output_probs:
            out = torch.softmax(out, dim=2)
            
        return out
