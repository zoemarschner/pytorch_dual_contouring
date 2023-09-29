import polyscope as ps

"""visualize using polyscope"""
def viz_mesh(V,F):
	ps.init()
	ps.register_surface_mesh("DC mesh", V, F)
	ps.show()


"""export to Wavefront obj format"""
def make_obj(file_name, V, F):
	with open(file_name, "w+") as file:
		for v in V:
			file.write("v {} {} {}\n".format(*v))

		for f in F:
			file.write("f {} {} {} {}\n".format(*f))



def flipYZ(V):
	V[:, [0,1]] = V[:,[1,0]]
	return V