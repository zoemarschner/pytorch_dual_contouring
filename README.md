# Pytorch Dual Contouring

The file `dual_contour.py` includes a function which computes a mesh of an implicit function using **Dual Contouring,**[^1] using pytorch's automatic differentiation to compute the gradients of the function. Using automatic differentiation for this makes dual contouring a drop-in replacement for marching cubes that will usually produce much higher quality meshes—especially for smaller grid sizes.

[^1]: Ju, Tao, Frank Losasso, Scott Schaefer, and Joe Warren. "Dual contouring of hermite data." In Proceedings of the 29th annual conference on Computer graphics and interactive techniques, pp. 339-346. 2002.
