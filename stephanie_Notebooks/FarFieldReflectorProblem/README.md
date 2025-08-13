# Solving the Far Field Reflector Problem 

## Background 

The "far field reflector problem" in  [geometric optics](https://en.wikipedia.org/wiki/Geometrical_optics)  involves designing a reflector surface that directs light from a source to a desired target distribution at a great distance  (in the far field). This is a type of **inverse problem**, where the desired outcome (the far-field pattern) is known, and the goal is to find the reflector shape that achieves it.

**Setup**: A point light source is placed at a known location, and the desired far-field radiation pattern is specified. This pattern typically describes how light intensity varies across a sphere at a large distance.

**Goal**:The objective is to find the shape of a reflector surface that will take the light emitted from the source and redirect it to match the specified far-field target distribution.

 When the reflector surface has a complex shape or when the desired far-field pattern is complex, this probalem can become very difficult, though we often often formulate it as an optimization problem.  One approach involves using the theory of optimal transport, where the reflector surface is found by minimizing the "transport cost" of light from the source to the target.  Here, we use the [Mong-Ampere equation](https://en.wikipedia.org/wiki/Monge%E2%80%93Amp%C3%A8re_equation)  to model the behavior of light rays as they reflect off the surface and  attempt to find the optimal mapping of light from the source to the target in order to determine the reflector shape. One example application of the Far Field Relfector Problem is in designing reflectors for antennas that focus radio waves in a desired direction.  

## Implementation

We implement  a neural network (NN) solution to the Near-Field [Reflector Problem](https://github.com/Ehsan494/Algorithms-for-convex-surface-reconstruction/blob/main/Selected%20Papers/MasMartinPatow_FastInverseReflectorDesign.pdf) using the [Monge-Kantorovich formulation](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)) of optimal transport (OT). The key goal is to find the optimal mapping between two 2D point clouds (sampled from 3D meshes), which can be interpreted physically as designing a reflector that maps one radiant source distribution onto a target distribution.

We depend on `cvxpy`,  `pytorch` for NNs and further  optimization, `polyscope` for 3D visualization, `trimesh` for mesh handling, as well as other scientific libraries.  Loading two 3D mesh files, we project  their vertices to 2D according to a selected view (`front`, `top`, or `side`), and normalizes and convert the points to PyTorch tensors for further processing. The two neural networks (Phi and Psi)  represent the dual Kantorovich potentials for the OT problem (source and target potentials) which we train using entropy-regularized OT loss, updating the NN parameters to maximize transport efficiency and  compute a cost matrix based on log of pairwise squared distances between points.  After training, we also compute the optimal transport plan (matrix), which encodes the probability of mapping each source point to each target point.  We incorporate checks by printing  stats about the cost matrix, optimal plan, and the overall OT cost for debugging and validation.

## Visualization 
Using the learned potential `phi` to  lift the source points to 3D, we construct the "reflector" shape and visualize these reflector points as a 3D point cloud.  In short, we taken two 3D shapes, projected them to 2D, learned an optimal mapping between their point clouds via neural OT, and visualized the resulting reflector geometry in 3D. 

---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
