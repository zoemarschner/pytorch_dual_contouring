"""
show example use of dual_contour.py to compute mesh for neural SDF
"""
import sys
sys.path.append('..')

from dual_contour import dual_contour
import torch
import polyscope as ps # for visualization
from utils import *

def square_3d(rs):
	## input is x,y,z as torch tensor
	def inner_func(inp):
		q = torch.abs(inp[:, :3]) - rs
		return torch.sqrt(torch.sum(torch.clamp(q, min=0)**2, dim=1))[:, None] + torch.clamp(torch.maximum(q[:,0],torch.maximum(q[:,1],q[:,2])),max=0)[:, None]

	return inner_func

if __name__ == "__main__":
	func = square_3d(torch.tensor([[0.3,0.3,0.3]]));
	V,F = dual_contour(func, N=5)
	
	viz_mesh(V,F)
	make_obj("./out/square.obj", V, F)


