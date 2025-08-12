# OT for 3D Shape Interpolation  

This notebook implements an interactive morphing between two 3D shapes (meshes) using [optimal transport (OT)](https://github.com/Ehsan494/Algorithms-for-convex-surface-reconstruction/blob/main/Selected%20Papers/Solomon_OT_DiscreteDomains.pdf) to compute a correspondence between their point clouds, a somewhat common method for shape interpolation in CG.  

In other words, we interpolate between the source and target shapes, and update the point cloud and visualization accordingly.
Given two sets of points `X` and `Y` (each Nx3), we compute an optimal transport plan (P) between them, where each point is assigned the same mass (uniform distribution)
and the **cost matrix** is the squared Euclidean distance between all pairs.  We use `cvxpy` to solve the [linear program](https://www.cvxpy.org/examples/basic/linear_program.html), returning the optimal transport matrix and its cost.

At `s=0:` You see the source with its original vertices and mesh.
As you increase `s` towards `1`: The points and mesh vertices move along straight lines (in 3D) toward their barycentric images computed by OT, allowing for smooth morphing.

At `s=1:` The mesh and point cloud are mapped as closely as possible (according to OT) to the target shape, but the mesh connectivity stays the same.

---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
