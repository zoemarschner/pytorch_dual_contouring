"""
show example use of dual_contour.py to compute mesh for neural SDF
"""
import sys
sys.path.append('..')
import time

from dual_contour import dual_contour
import torch
import polyscope as ps # for visualization
from utils import *


def square_3d(rs):
	## input is x,y,z as torch tensor
	def inner_func(inp):
		q = torch.abs(inp[:, :3]) - rs

		side = torch.sqrt(torch.sum(torch.where(q > 0, q**2, 0), dim=1))[:, None]
		corner = torch.clamp(torch.maximum(q[:,0],torch.maximum(q[:,1],q[:,2])),max=0)[:, None]
		return  side + corner

	return inner_func

if __name__ == "__main__":
	N = 15
	func = square_3d(torch.tensor([[0.3,0.3,0.3]]));

	compare_dc_mc(func, N=N, save_name="cube", VIZ=True)