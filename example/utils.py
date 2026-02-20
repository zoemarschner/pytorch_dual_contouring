import polyscope as ps
import mcubes 
import torch, time
from dual_contour import dual_contour
import numpy as np

"""visualize using polyscope"""
def viz_mesh(V,F,name="mesh"):
	ps.init()
	ps.register_surface_mesh("DC mesh", V, F)
	ps.register_point_cloud('pts', V)
	ps.show()

def compare_dc_mc(f, N=50, domain_min=[-0.5,-0.5,-0.5], domain_max=[0.5,0.5,0.5], save_name=None, VIZ=True, device='cpu'):
	start = time.time()
	V,F = dual_contour(f, N=N, domain_min=domain_min, domain_max=domain_max, device=device)
	end = time.time()
	print(f'-> FINISHED DUAL CONTOURING IN {end-start} SECONDS')
	
	if VIZ:
		viz_mesh(V,F,"DC mesh")
	if save_name:
		make_obj(f"./out/{save_name}_{N}_dc.obj", V, F)

	start = time.time()
	V,F = marching_cubes(f, N=N, domain_min=domain_min, domain_max=domain_max)
	end = time.time()
	print(f'-> FINISHED MARCHING CUBES IN {end-start} SECONDS')
	
	if VIZ:
		viz_mesh(V,F, "MC mesh")
	if save_name:
		make_obj(f"./out/{save_name}_{N}_mc.obj", V, F)


"""export to Wavefront obj format"""
def make_obj(file_name, V, F):
	# check if F starts at 0, if it does add 1
	if F.min() == 0:
		F = F + 1

	with open(file_name, "w+") as file:
		for v in V:
			file.write("v {} {} {}\n".format(*v))

		for f in F:
			file.write("f {} {} {}\n".format(*f))

def flipXY(V):
	V[:, [0,1]] = V[:,[1,0]]
	return V

def flipYZ(V):
	V[:, [1,2]] = V[:,[2,1]]
	return V

# wrapper around mcubes library for maching cubes implementation to compare to
def marching_cubes(f, N=50, domain_min=[-0.5,-0.5,-0.5], domain_max=[0.5,0.5,0.5]):
	if isinstance(N, int):
		N = (N,N,N)

	ranges = [torch.linspace(bot, up, n) for bot, up, n in zip(domain_min, domain_max, N)]
	X,Y,Z = torch.meshgrid(*ranges, indexing="ij")
	grid_pts = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)

	vals = -f(grid_pts)
	vals = vals.reshape(X.shape).detach().numpy()

	V,F = mcubes.marching_cubes(vals, 0)
	V = V/np.array(N)

	tmin = np.array(domain_min); tmax = np.array(domain_max)
	V = V * (tmax-tmin)+tmin
	
	return (V,F)


