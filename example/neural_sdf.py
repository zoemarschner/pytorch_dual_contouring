import sys
sys.path.append('..')

import torch
from torch import nn
from dual_contour import dual_contour
from utils import *

class DeepSDF(nn.Module):
    def __init__(self, num_layers, hidden_size, n_dim=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_dim, hidden_size), nn.ReLU()])
        
        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_size, 1))
                
    def forward(self, features):
        out = features
        
        for layer in self.layers:
            out = layer(out)        
        return out

if __name__ == "__main__":
    model = DeepSDF(8, 128)
    model.load_state_dict(torch.load('./data/neural_sdf_union.pt'))
    
    V,F = dual_contour(model, N=32, domain_min=[-2,-2,-2], domain_max=[2,2,2])
    V = flipYZ(V)

    viz_mesh(V,F)
    make_obj("./out/square.obj", V, F)


   