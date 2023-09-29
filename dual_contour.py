import torch
import numpy as np

"""
returns a mesh of *f* computed using dual contouring.
If f_grad is not provided, it will be calculated with pytorch autodiff
"""
def dual_contour(f, f_grad=None, N=50, domain_min=[-0.5,-0.5,-0.5], domain_max=[0.5,0.5,0.5], device="cpu"):
	# build grid
	print('hi1....')

	if isinstance(N, int):
		N = (N,N,N)

	ranges = [torch.linspace(bot, up, n, device=device) for bot, up, n in zip(domain_min, domain_max, N)]
	X,Y,Z = torch.meshgrid(*ranges, indexing="xy")
	grid_pts = torch.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

	# evaluate function at the grid points
	f_vals = f(grid_pts)

	# calculate vertices
	tmin = torch.tensor(domain_min, device=device); tmax = torch.tensor(domain_max, device=device)
	dc_vertices = grid_pts + (tmax - tmin)*1/torch.tensor(N, device=device) * 0.5


	v_i = lambda x,y,z: x*N[1]*N[2] + y*N[2] + z

	# compute faces based on sign changes
	f_vals_grid = f_vals.reshape(X.shape)

	make_face = lambda i1,i2,i3,i4,swp: [v_i(*i4), v_i(*i3), v_i(*i2), v_i(*i1)] if swp else [v_i(*i1), v_i(*i2), v_i(*i3), v_i(*i4)]
	print('hi2....')

	for x in range(N[0]-1):
		for y in range(N[1]-1):
			for z in range(N[2]-1):
				swp = f_vals_grid[x,y,z]>0
				if f_vals_grid[x,y,z]*f_vals_grid[x,y,z+1] < 0:
					faces.append(make_face((x-1, y, z),(x, y, z),(x, y-1, z),(x-1, y-1, z), not swp))

				if f_vals_grid[x,y,z]*f_vals_grid[x,y+1,z] < 0:
					faces.append(make_face((x, y, z-1), (x, y, z), (x-1, y, z), (x-1, y, z-1), not swp))

				if f_vals_grid[x,y,z]*f_vals_grid[x+1,y,z] < 0:
					faces.append(make_face((x, y-1, z-1), (x, y-1, z), (x, y, z), (x, y, z-1),not swp))
	faces = np.vstack(faces)+1
	print('hi3....')


	return dc_vertices.detach().numpy(), faces.numpy()

