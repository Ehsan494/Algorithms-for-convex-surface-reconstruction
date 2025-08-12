# Solving the Near Field Reflector Problem 

## Theory 
We implement  a neural network (NN) solution to the Near-Field Reflector Problem using the Monge-Kantorovich formulation of optimal transport (OT). The key goal is to find the optimal mapping between two 2D point clouds (sampled from 3D meshes), which can be interpreted physically as designing a reflector that maps one radiant source distribution onto a target distribution.

We depend on cvxpy,  PyTorch for NNs and further  optimization, Polyscope for 3D visualization, Trimesh for mesh handling, as well as other scientific libraries.  Loading two 3D mesh files, we project  their vertices to 2D according to a selected view (`fron`t, `top`, or `side`), and normalizes and convert the points to PyTorch tensors for further processing. The two neural networks (Phi and Psi)  represent the dual Kantorovich potentials for the OT problem (source and target potentials) which we train using entropy-regularized OT loss, updating the NN parameters to maximize transport efficiency and  compute a cost matrix based on log of pairwise squared distances between points.  After training, we also compute the optimal transport plan (matrix), which encodes the probability of mapping each source point to each target point.  We incorporate checks by printing  stats about the cost matrix, optimal plan, and the overall OT cost for debugging and validation.

## Visualization 
Using the learned potential `phi` to  lift the source points to 3D, we construct the "reflector" shape and visualize these reflector points as a 3D point cloud.  In short, we taken two 3D shapes, projected them to 2D, learned an optimal mapping between their point clouds via neural OT, and visualized the resulting reflector geometry in 3D. 

---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
