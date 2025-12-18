import torch
import sys
from pathlib import Path

# Add NeurASP to path
current_dir = Path(__file__).resolve().parent
neurasp_path = current_dir.parent.parent / "vendor/NeurASP"
sys.path.append(str(neurasp_path))

from neurasp import NeurASP
from src.models.neural import SortingNetwork

class SymbolicSorter:
    def __init__(self, device="cpu", lr=0.001):
        self.device = device
        
        # Load ASP Rules
        asp_path = current_dir.parent / "asp" / "sort.lp"
        with open(asp_path, 'r') as f:
            base_asp = f.read()
            
        # Define Neural Rule
        # nn(neural_pos(5, vals), (0,1,2,3,4))
        # This creates atoms like neural_pos(0, vals, 0), neural_pos(0, vals, 1)...
        
        self.dprogram = """
        % Neural Rule
        % 5 entities (indices 0-4), input term 'vals', domain (0-4 positions)
        nn(neural_pos(5, vals), (0,1,2,3,4)).
        
        % Mapping
        position(I, P) :- neural_pos(I, vals, P).
        
        """ + base_asp
        
        self.model = SortingNetwork(input_size=5)
        self.model.set_output_probs(True)
        self.nn_mapping = {"neural_pos": self.model}
        self.optimizers = {"neural_pos": torch.optim.Adam(self.model.parameters(), lr=lr)}
        
        self.neurasp = NeurASP(self.dprogram, self.nn_mapping, self.optimizers, gpu=(device!="cpu"))
        
    def learn(self, data_list, obs_list, epochs=10, alpha=0.5, batch_size=32):
        self.neurasp.learn(dataList=data_list, obsList=obs_list, epoch=epochs, alpha=alpha, batchSize=batch_size, bar=True)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
