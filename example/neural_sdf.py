import sys
sys.path.append('..')

import torch
from torch import nn
from dual_contour import dual_contour
from utils import *
import time 
import mcubes

VIZ = False

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
	device = 'cpu'
	N=128

	model = DeepSDF(5, 64).to(device)
	model.load_state_dict(torch.load('./data/rockerarm.pt'))
	
	compare_dc_mc(model, N=N, domain_min=[-2,-2,-2], domain_max=[2,2,2], save_name="rockerarm", VIZ=True, device=device)


   