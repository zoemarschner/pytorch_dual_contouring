# Pytorch Dual Contouring

<p align="center">
<img src="example/comaparison_image.png" alt="Comparison of dual contouring and marching cubes" width="500">
</p>

The file `dual_contour.py` includes a function which computes a mesh of an implicit function with **Dual Contouring,**[^1] using pytorch's automatic differentiation to compute the gradient of the function. Using automatic differentiation for this makes dual contouring a drop-in replacement for marching cubes that will usually yield much nicer quality meshes—especially for smaller grid sizes.

## Examples

The `example/` folder includes two examples uses of the dual contouring script on SDFs defined in different ways: `analytic_sdf.py` on an [analytically defined SDF](https://iquilezles.org/articles/distfunctions/) and `neural_sdf.py` on a pre-trained neural SDF (example neural SDF weights are included in the `example/data` folder).

[^1]: Dual contouring was first introducedd in Ju, Tao, Frank Losasso, Scott Schaefer, and Joe Warren. "Dual contouring of hermite data." In Proceedings of the 29th annual conference on Computer graphics and interactive techniques, pp. 339-346. 2002.
