import torch
import numpy as np
from enum import Enum

LST_SQ_DRIVER = 'gelsy'

GradCompMethod = Enum('GradCompMethod', ['AUTOGRAD', 'FINITE_DIFF'])
ClipMethod = Enum('ClipMethod', ['NONE', 'CLIP', 'SOLVE_BD_PROB'])

"""
Returns a mesh of *f* computed using dual contouring.
Arguments:
	f: the function to dual contour the zero level set of; takes in Nx3 points and returns 
		corresponds Nx1 values
Optional arguments:
	f_grad: if set, this function will be used to compute the gradients of f; takes in Nx3 points
		and returns Nx3 corresponding gradients. If none, gradient is automatically computed
	N: grid size for dual contouring, either integer or 3 tuple
	domain_min, domain_max: 3 tuples specifying minimum and maximum of domain 
		over which to dual contour
	grad_comp: an element of GradCompMethod enum which determines how to automatically
		compute the gradient if f_grad is None (either with autograd or finite difference)
	clip_method: an element of ClipMethod enum which determines how to deal with solves that 
		fall outside the proper grid cell. options:
		NONE: don't do anything, will often result in spike artifacts in reconstruction (not reccomended)
		CLIP: clips any values outside cell to lie in cell
		SOLVE_BD_PROB: solves a sequence of constrained optimization problems and determines 
			which provides best reconstruction for the cell (default, reccomended)
	bias: the amount of bias to add to encorage good solutions (set to zero to remove bias)
	device: device to do computation on, should match with device f uses
"""
def dual_contour(f, f_grad=None, N=50, domain_min=[-0.5,-0.5,-0.5], domain_max=[0.5,0.5,0.5], grad_comp=GradCompMethod.AUTOGRAD, clip_method=ClipMethod.SOLVE_BD_PROB, bias=0.3, device="cpu"):
	# process input
	if isinstance(N, int):
		N = (N,N,N)

	# calculate deltas in flat indices for each dimension of this array
	flat_ds = [N[2]*N[1], N[2], 1]

	if f_grad is None:
		if grad_comp == GradCompMethod.AUTOGRAD:
			f_grad = pytorch_autodiff_gradients(f)
		elif grad_comp == GradCompMethod.FINITE_DIFF:
			f_grad = grad_finite_diff(f)
		else:
			raise NotImplementedError("Unsupported gradient computation method!")
	
	# build grid
	ranges = [torch.linspace(bot, up, n, device=device) for bot, up, n in zip(domain_min, domain_max, N)]
	X,Y,Z = torch.meshgrid(*ranges, indexing="ij")
	grid_pts = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

	# evaluate function at the grid points
	f_vals = f(grid_pts)


	# =========== COMPUTE EDGE LOCATIONS VIA SIGN CHANGES ===========

	f_vals_grid = f_vals.reshape(X.shape)

	# find edges crossing zero -> faces in object
	x_edges_diff = (f_vals_grid[:-1,:,:]*f_vals_grid[1:,:,:] < 0)
	y_edges_diff = (f_vals_grid[:,:-1,:]*f_vals_grid[:,1:,:] < 0)
	z_edges_diff = (f_vals_grid[:,:,:-1]*f_vals_grid[:,:,1:] < 0)
	edge_diffs = (x_edges_diff, y_edges_diff, z_edges_diff)

	edges = [edge_diff.nonzero() for edge_diff in edge_diffs]
	x_edges, y_edges, z_edges = edges

	# flatten to indices into vertex array
	flat = [_flatten_index(i_edges, N) for i_edges in edges]
	x_flat, y_flat, z_flat = flat

	# ============== COMPUTE OPTIMAL VERTEX LOCATIONS ===============

	# calculate grid spacing
	tmin = torch.tensor(domain_min, device=device); tmax = torch.tensor(domain_max, device=device)
	tN = torch.tensor(N, device=device)
	spacing = (tmax - tmin)*1/tN
	locs = [grid_pts[i_flat, :] for i_flat in flat]

	# linearly interpolated position where f equals 0
	for i, (i_locs, i_sp, i_flat, di) in enumerate(zip(locs, spacing, flat, flat_ds)):
		i_locs[:, i] += i_sp*(-f_vals[i_flat]/(f_vals[i_flat+di]-f_vals[i_flat])).squeeze()

	# compute grads at that location
	total_locs = torch.cat(locs, dim=0)
	total_grads = f_grad(total_locs).detach()
	total_locs = total_locs.detach() # detach in case it was used with autograd in f_grad 

	# idenitfy locations corresponding for cube edges for each cube point
	xyz_i = (f_vals_grid[:-1,:-1,:-1]*0 + 1).nonzero().unsqueeze(2) # (everything is 0, this just gets indices)

	cube_locs = [xyz_i.repeat(1,1,4) for _ in range(3)]

	cube_locs_x, cube_locs_y, cube_locs_z = cube_locs
	cube_locs_x[:,[1,2,1,2],[1,1,2,3]] += 1
	cube_locs_y[:,[0,2,0,2],[1,1,2,3]] += 1
	cube_locs_z[:,[0,1,0,1],[1,1,2,3]] += 1

	# calculate mask that identifies which edges cross zero for each cube point
	zero_edge_mask = [edge_diff[cube_locs_i.chunk(chunks=3, dim=1)] for (edge_diff, cube_locs_i) in zip(edge_diffs, cube_locs)]

	# calculate how many total edge=0 crossings each cube location has
	cube_loc_N = sum([mask.sum(dim=(1,2)) for mask in zero_edge_mask])

	# create index grids, which map between x,y,z locations and indices into total_locs/total_grads
	index_grids = [-1*torch.ones(i_edges_diff.shape, dtype=torch.int, device=device) for i_edges_diff in edge_diffs]
	num_edges = [i_edges_diff.sum() for i_edges_diff in edge_diffs]

	for i, (ind_grid, i_edges_diff) in enumerate(zip(index_grids, edge_diffs)):
		# add a row of zeros to the correct dimension to index into overall grid
		offset = sum(num_edges[:i])
		ind_grid[i_edges_diff] = torch.arange(num_edges[i], dtype=torch.int, device=device) + offset

	# store calculated dc vertices here
	# dc_vertices = grid_pts + (tmax - tmin)*1/torch.tensor(N, device=device) * 0.5
	dc_vertices = torch.zeros(grid_pts.shape, device=device)
	dc_vertices_mask = torch.zeros((grid_pts.shape[0]), dtype=torch.bool, device=device)

	# now, we will seperate the cubes into slices that have the same total number of zero crossings
	# since this determines the size of the least squares problem that should be solved,
	# this allows us to batch solve the least squares problem	
	for k in range(1, max(cube_loc_N)+1):
		k_mask = (cube_loc_N == k)

		if (Nk := k_mask.sum()) == 0:
			continue

		# find indices into array of positions/normals for the problems with k positions
		cube_locs_k = [cube_locs_i[k_mask, :, :] for cube_locs_i in cube_locs]
		edge_indices_k = [ind_grid[cube_locs_i.chunk(chunks=3, dim=1)].squeeze() for ind_grid,cube_locs_i in zip(index_grids, cube_locs_k)]
		edge_indices = torch.cat(edge_indices_k, dim=1)
		edge_indices = edge_indices[(edge_indices+1).nonzero().chunk(chunks=2, dim=1)].reshape((Nk, k))

		k_positions = total_locs[edge_indices, :]
		k_normals 	= total_grads[edge_indices, :]

		flat_cell_ind = _flatten_index(cube_locs_k[0][:, :, 0], N)

		v_opt = _solve_lst_sq(k_positions, k_normals, grid_pts[flat_cell_ind], spacing, bias=bias, clip_method=clip_method)

		# map optimal vertex computation points back to array for final joining stpe
		dc_vertices[flat_cell_ind] = v_opt
		dc_vertices_mask[flat_cell_ind] = 1

	# ======================= CONSTRUCT FACES =======================

	# construct faces
	dx, dy, dz = flat_ds
	z_faces = torch.vstack((z_flat-dx, z_flat, z_flat-dy, z_flat-(dx+dy))).T # (x-1, y, z),(x, y, z),(x, y-1, z),(x-1, y-1, z)
	y_faces = torch.vstack((y_flat-dz, y_flat, y_flat-dx, y_flat-(dx+dz))).T # (x, y, z-1), (x, y, z), (x-1, y, z), (x-1, y, z-1)
	x_faces = torch.vstack((x_flat-(dy+dz), x_flat-dy, x_flat, x_flat-dz)).T # (x, y-1, z-1), (x, y-1, z), (x, y, z), (x, y, z-1)
	faces = (x_faces, y_faces, z_faces)

	# swap so that all faces point correct direction
	for i_faces, i_edges in zip(faces, edges):
		swp = (f_vals_grid[i_edges.chunk(chunks=3, dim=1)] < 0).flatten()
		i_faces[swp,:] = i_faces[swp,:].flip(dims=(1,))

	faces = torch.vstack(faces)

	# filter out zeros values of dc_vertices using pre-computed mask...
	filt_indices = torch.zeros(dc_vertices.shape[0], device=dc_vertices.device, dtype=torch.long)
	filt_indices[dc_vertices_mask] = torch.arange(torch.count_nonzero(dc_vertices_mask))
	filt_F = filt_indices[faces]
	filt_V = dc_vertices[dc_vertices_mask,:]

	return filt_V.cpu().numpy(), _triangulate(filt_V,filt_F).cpu().numpy()

# ==============================================================
# ===============        HELPER FUNCTIONS        ===============
# ==============================================================

"""
solve the linear least squares problem for where to put grid points,
given batched arrays of positions and corresponding positions
other argments are required information/settings passed on from calling function
	cell_pts: bottom point of the cell, 
	spacing: size of cell in each dimenstion
	bias: amount of bias towards good solutions to add
	clip_method: method for dealing with invalid locations
these two variables are needed for the methods which ensure the solution lies in the cell
"""
def _solve_lst_sq(positions, normals, cell_pts=None, spacing=None, bias=0, clip_method=ClipMethod.NONE):
	Nk, k, _ = positions.shape

	# add bias to linear solve to encourage point being 
	if bias != 0:
		center_pt = positions.sum(dim=1)/k
		bias_normals = torch.tensor([[bias, 0, 0], [0, bias, 0], [0, 0, bias]], device=positions.device)
		bias_normals = bias_normals.unsqueeze(0).repeat((Nk,1,1))

		bias_positions = center_pt.unsqueeze(1).repeat((1,3,1))

		positions = torch.cat((positions, bias_positions), dim=1)
		normals = torch.cat((normals, bias_normals), dim=1)

	# calculate A and b for least squares
	A = normals
	b = torch.einsum('ijk, ijk -> ij', A, positions).unsqueeze(2)
	
	v_opt = torch.linalg.lstsq(A, b, driver=LST_SQ_DRIVER).solution.squeeze()

	# certain linear solves can lead to cases where the best solution can be in many
	# positions, leading to solutions far outside the box, which causes spikes in the model
	# these next two techniques try to prevent that.

	# if SOLVE_BD_PROB is clip method, solve probelm constrained to boundary of decreasing dimension
	# when solution is outside the box, to gurantee a good feasible solution
	if clip_method == ClipMethod.SOLVE_BD_PROB:
		# generator for plane constraints
		plane_gen = (([side], [dim]) for side in (0,1) for dim in range(3))
		v_opt = _solve_fixed_problems(A, b, v_opt, cell_pts, spacing, plane_gen)

		# generator for line constraints
		edge_gen = (([side1,side2], [dim1,dim2]) for side1 in (0,1) for dim1 in range(3) for side2 in (0,1) for dim2 in range(dim1+1,3))
		v_opt = _solve_fixed_problems(A, b, v_opt, cell_pts, spacing, edge_gen)

		# generator for vertex contraints
		point_gen = (([side1,side2,side3], [0,1,2]) for side1 in (0,1) for side2 in [0,1] for side3 in [0,1])
		v_opt = _solve_fixed_problems(A, b, v_opt, cell_pts, spacing, point_gen)

	# a more naive method is to simply clip to lie in the grid cell
	if clip_method == ClipMethod.CLIP:
		deltas = (v_opt - cell_pts)/spacing
		print(f'CLIPPED SOLUTION FOR {torch.any(torch.logical_or(deltas < 0, deltas > 1), dim=1).sum()} CELLS')
		clip_deltas = torch.clip(deltas, min=0, max=1)

		v_opt = clip_deltas*spacing + cell_pts


	return v_opt



"""
solve a series of least square problems with A and b, fixing the 
locations specified by possiblities_gen for all values where provided v_opt
does not lie in the grid cell, and finding which of the provided fixed locations provides
the lowest error among these.
	possiblities_gen is a generator that yields tuples of (sides, dims)
	v_opt, cell_pts, spacing are passed along values nessecary for operation
returns entire v_opt with new locations replacing previously incorrect locations. 
"""
def _solve_fixed_problems(A, b, v_opt, cell_pts, spacing, possiblities_gen):
	# find problem cases
	Af, bf, mask = _filter_invalid_solutions(A, b, v_opt, cell_pts, spacing) 

	# constrain to planes
	fixed_sol = torch.zeros((Af.shape[0],3,1), device=A.device)
	cur_errors  = float('inf') * torch.ones(Af.shape[0], device=A.device)

	for sides, dims in possiblities_gen: # constrain to plane at beginning of cube or end?
		sol, err = _lst_sq_fix_dimesions(Af, bf, dims, sides, cell_pts[mask,:], spacing)
		replace_mask = err < cur_errors
		cur_errors[replace_mask] = err[replace_mask]
		fixed_sol[replace_mask, :] = sol[replace_mask, :]

	v_opt[mask, :] = fixed_sol.squeeze()

	return v_opt

"""
filter the problems Ak and bk, with precomputed solutions sols, to only contain the problems 
that lead to points outside the grid cells defined by cell_pts and spacing
returns a tuple of the filtered Ak and bk and the mask for recombining the solutions
"""
def _filter_invalid_solutions(A, b, sols, cell_pts, spacing):
	deltas = (sols - cell_pts)/spacing
	mask = torch.any(torch.logical_or(deltas < 0, deltas > 1), dim=1)

	return A[mask, :, :], b[mask, :, :], mask


"""
solves the least squares problem with the coordinates in dimension dims
fixed to be equal to cell_pts + spacing*sides
returns tuple of fixed solution and associated errors
NOTE: assumes that dims is in ascending order!!
"""
def _lst_sq_fix_dimesions(A, b, dims, sides, cell_pts, spacing):
	# adjust b to use the fixed value
	values = []
	for dim, side in zip(dims,sides):
		value = (cell_pts + spacing*side)[:,dim].unsqueeze(1)
		values.append(value)
		b = b - (A[:,:,dim] * value).unsqueeze(2)

	if len(dims) != A.shape[2]:
		for i, dim in enumerate(dims):
			# remove rows from A
			A = torch.cat((A[:,:,:(dim-i)],A[:,:,(dim-i+1):]), dim=2)

		v_opt = torch.linalg.lstsq(A, b, driver=LST_SQ_DRIVER).solution

		errors = torch.sqrt(((A@v_opt - b)**2).sum(dim=1)).squeeze(dim=1)

		fixed_sol = v_opt
		for dim, value in zip(dims, values):
			fixed_sol = torch.cat((fixed_sol[:,:dim,:], value.unsqueeze(2), fixed_sol[:, dim:,:]), dim=1)
	else:
		errors = torch.sqrt(((b)**2).sum(dim=1)).squeeze(dim=1)
		fixed_sol = torch.cat(values, dim=1).unsqueeze(2)

	return fixed_sol, errors


"""
given an array of indices into a 3D array of size N, compute the corresponding
indices of the flattened array
"""
def _flatten_index(indices, N):
	return indices[:,0]*N[1]*N[2] + indices[:,1]*N[2] + indices[:,2]


# given array F (Nx4) of quadrilaterals, naivly divide them into triangles (2Nx3)
# method: compute area of triangles in both edge flip possiblities, pick minimum of the two
def _triangulate(V, F):

	# tris = reshape(X(T, :), [size(T, 1), 3, 3]);
	# crosses = cross(tris(:,2,:) - tris(:,1,:), tris(:,3,:) - tris(:,1,:));
	# areas = 1/2*sqrt(sum(crosses.^2, 3));
	# A = sum(areas, 'all');
	tri_opts = ([0,1,3], [1,2,3], [0,1,2], [0,2,3])
	ts = [F[:, tri] for tri in tri_opts]
	opt_areas = []
	for t in ts:
		t_pts = V[t,:]
		crosses = torch.cross(t_pts[:,1,:]-t_pts[:,0,:], t_pts[:,2,:]-t_pts[:,0,:])
		areas = 1/2 * torch.sqrt((crosses**2).sum(dim=1))
		opt_areas.append(areas)

	subdiv_2_best = sum(opt_areas[2:]) > sum(opt_areas[:2]) 

	top_tris = ts[0]
	top_tris[subdiv_2_best, :] = ts[2][subdiv_2_best, :]

	bot_tris = ts[1]
	bot_tris[subdiv_2_best, :] = ts[3][subdiv_2_best, :]


	return torch.cat((top_tris,bot_tris), dim=0)


# ==============================================================
# ==============       GRADIENT FUNCTIONS       ================
# ==============================================================

"""
simple implementation of centered finite difference which returns a function
which computes the gradient (Nx3) of f at a tensor of Nx3 input points
"""
def grad_finite_diff(f, delta=0.0001):
	def inner(pts):
		dx,dy,dz = [delta,0,0], [0,delta,0], [0,0,delta]

		grads = []
		for d in [dx,dy,dz]:
			arr_d = torch.tensor([d], device=pts.device)
			grads.append((f(pts + arr_d)-f(pts - arr_d))/(2*delta))

		return torch.cat(grads, dim=1)
		
	return inner


def pytorch_autodiff_gradients(input_func):
	def inner(pts):
		pts_var = torch.autograd.Variable(pts.data, requires_grad=True)
		vals = input_func(pts_var)

		grad_total, = torch.autograd.grad(vals, pts_var, grad_outputs=torch.ones(vals.shape, device=pts.device))
		graddx = grad_total[:,:3]

		return graddx

	return inner

