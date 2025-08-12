
# NN implmentation of the Near-Field Reflector problem 
# using the Monge-Kantorovich formulation of optimal transport/

import torch
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
import trimesh
import gpytoolbox as gpy
from pathlib import Path
from tqdm import tqdm 
import polyscope as ps


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sample point clouds 
def sample_cloud(mesh_file, view='front'):
    """
    Load and project 3D mesh to 2D.
    
    Args:
        mesh_file: Path to mesh file
        N: Number of vertices (used for compatibility)
        view: 'front' (xy), 'top' (xz), or 'side' (yz)
    """
    # Resolve mesh_file to an absolute path relative to this script
    mesh_path = Path(mesh_file)
    if not mesh_path.is_absolute():
        mesh_path = Path(__file__).parent / mesh_path
        
    mesh = trimesh.load(str(mesh_path), process=False)
    
    # Rescale mesh vertices to fit in [-1, 1]^3 cube, centered at origin
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # Center at origin
    vertices -= np.mean(vertices, axis=0)
    # Scale to fit in [-1, 1]
    vertices /= np.max(np.abs(vertices))
    
    # Project to 2D based on view
    if view == 'front':
        vertices_2d = vertices[:, :2]
    elif view == 'top':
        vertices_2d = vertices[:, [0, 2]]  # x, z
    elif view == 'side':
        vertices_2d = vertices[:, [2, 1]]  # y, z
    else:
        raise ValueError("view must be 'front', 'top', or 'side'")
    
    vertices_2d = vertices_2d.astype(np.float32)
    
    return vertices_2d, faces.astype(np.int32)

# initialize the NNs for the dual potentials functions 
class Phi(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Phi, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Psi(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Psi, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# define input dimensions
Ds = 2  # source dimension (2D points)
Dt = 2  # target dimension (2D points)

# define the source and target potential functions
phi = Phi(Ds).to(device)
psi = Psi(Dt).to(device)
opt = torch.optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=1e-3)

epochs = 1000
# the epsilon parameter controls the entropy regularization strength
# smaller epsilon means more regularization, larger epsilon means less regularization
epsilon = 0.005

# choose orientation for projection: 'front', 'top', 'side'
source_vertices, source_faces = sample_cloud('data/bunny.obj', view='side')
target_vertices, target_faces = sample_cloud('data/spot.obj', view='front')

# sample points from each point cloud
X = source_vertices
Y = target_vertices

# normalize X and Y to unit vectors
X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-12)

# Convert to torch tensors
X = torch.as_tensor(X, dtype=torch.float32, device=device)
Y = torch.as_tensor(Y, dtype=torch.float32, device=device)

for epoch in tqdm(range(epochs), desc="Training OT Model"):
    X_exp = X[:, None, :]  # (num_source, 1, d)
    Y_exp = Y[None, :, :]  # (1, num_target, d)
    
    sq_dist = torch.sum((X_exp - Y_exp) ** 2, dim=-1)  # (num_source, num_target)
    sq_dist = sq_dist.clamp(min=1e-12)  # avoid log(0)
    C = -torch.log(0.5 * sq_dist)
    C = C.clamp(min=0)  # ensure cost is non-negative

    P = phi(X).squeeze(-1)  # shape (num_source,)
    S = psi(Y).squeeze(-1)  # shape (num_target,)
    
    P = P[:, None] # shape (num_source, 1)
    S = S[None, :] # shape (1, num_target)

    Z = torch.exp((P + S - C)/epsilon)

    L_entropy = epsilon * (Z.mean() - 1.0) - P.mean() - S.mean()

    opt.zero_grad()
    L_entropy.backward()
    opt.step()

# display the output
P_opt = Z / Z.sum(dim=1, keepdim=True)

# find the maximum pairs for phi and psi
phi_max_pairs = P_opt.argmax(dim=0)
psi_max_pairs = P_opt.argmax(dim=1)

# visualize reflector as a 3D point cloud
with torch.no_grad():
    # if X is 2D, lift to 3D
    if X.shape[1] == 2:
        X3d = torch.cat([X, torch.zeros(X.shape[0], 1, device=X.device)], dim=1)
    else:
        X3d = X
    # compute the reflector points according to the phi function: exp(phi(sigma)) * X3d
    phi_vals = phi(X).squeeze(-1)
    reflector_points = torch.exp(phi_vals)[:, None] * X3d
    reflector_points_np = reflector_points.cpu().numpy()

ps.init()
ps.set_ground_plane_mode("none")
ps_cloud = ps.register_point_cloud("reflector 3D points", reflector_points_np)
ps_cloud.set_radius(0.01)
ps_cloud.set_point_render_mode("sphere")
ps_cloud.set_color((0.9, 0.6, 0.1))
print("Reflector 3D point cloud visualized in Polyscope.")
ps.show()

print("--- Sanity checks ---")
print(f"C: min={C.min().item():.4f}, max={C.max().item():.4f}, mean={C.mean().item():.4f}")
print(f"P_opt: min={P_opt.min().item():.4f}, max={P_opt.max().item():.4f}, sum={P_opt.sum().item():.4f}")
print(f"C shape: {C.shape}, P_opt shape: {P_opt.shape}")
print(f"Any NaN in C? {torch.isnan(C).any().item()} | Any Inf in C? {torch.isinf(C).any().item()}")
print(f"Any NaN in P_opt? {torch.isnan(P_opt).any().item()} | Any Inf in P_opt? {torch.isinf(P_opt).any().item()}")

cost = (P_opt * C).sum()
print(f"OT cost: {cost.item():.4f}")
