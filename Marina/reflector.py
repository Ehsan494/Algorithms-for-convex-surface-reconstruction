# NN implmentation of the Near-Field Reflector problem 
# using the Monge-Kantorovich formulation of optimal transport

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# sample point clouds 
def sample_cloud(mesh_file, plane='xy', coord=0.0):
    """Load mesh, normalize to [-1,1]^3, project onto a coordinate plane at a fixed face.

    Parameters
    ----------
    mesh_file : str or Path
        Mesh path.
    plane : {'xy','xz','yz'}
        Plane onto which to project (orthographic flattening). Previous views
        'front','top','side' map to 'xy','xz','yz' respectively. Use different
        planes for perpendicular initialization.
    coord : float
        Fixed coordinate value for the dropped axis. Use +1 or -1 to place the
        point cloud on a face of the unit cube ("perpendicular initialization").

    Returns
    -------
    (points, faces)
        points : (N,3) float32
        faces  : (F,3) int32 (unchanged indices from original mesh)
    """
    mesh_path = Path(mesh_file)
    if not mesh_path.is_absolute():
        mesh_path = Path(__file__).parent / mesh_path

    mesh = trimesh.load(str(mesh_path), process=False)

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # Normalize to centered unit cube
    vertices -= np.mean(vertices, axis=0)
    max_abs = np.max(np.abs(vertices))
    if max_abs > 0:
        vertices /= max_abs

    if plane == 'xy':
        vertices_out = np.stack([vertices[:,0], vertices[:,1], np.full_like(vertices[:,0], coord)], axis=1)
    elif plane == 'xz':
        vertices_out = np.stack([vertices[:,0], np.full_like(vertices[:,0], coord), vertices[:,2]], axis=1)
    elif plane == 'yz':
        vertices_out = np.stack([np.full_like(vertices[:,0], coord), vertices[:,1], vertices[:,2]], axis=1)
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    return vertices_out.astype(np.float32), faces.astype(np.int32)

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
    
# define input dimensions (flattened 3D coordinates lying in a plane)
Ds = 3  # source dimension (3D flattened)
Dt = 3  # target dimension (3D flattened)

# define the source and target potential functions
phi = Phi(Ds).to(device)
psi = Psi(Dt).to(device)
opt = torch.optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=1e-3)

epochs = 1000
# Entropy regularization annealing:
# Start with a larger epsilon for stability / exploration, decay to a small value for sharper transport.
epsilon_start = 0.05
epsilon_final = 0.001
anneal_portion = 0.8  # fraction of epochs over which to decay

def epsilon_annealing(epoch: int):
    total_decay_epochs = int(anneal_portion * epochs)
    if total_decay_epochs <= 0:
        return epsilon_final
    if epoch >= total_decay_epochs:
        return epsilon_final
    t = epoch / total_decay_epochs  # in [0,1)
    # Exponential (geometric) decay: eps = start * (final/start)^t
    return float(epsilon_start * (epsilon_final / epsilon_start) ** t)

source_vertices, source_faces = sample_cloud('data/bunny.obj', plane='xy', coord=1.0)
target_vertices, target_faces = sample_cloud('data/spot.obj', plane='yz', coord=1.0)

# sample points from each point cloud (original full sets)
X = source_vertices + np.ones((502, 3))
Y = target_vertices + 2*np.ones((3225, 3))

# normalize X and Y to unit vectors
X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-12)

# Convert to torch tensors
X = torch.as_tensor(X, dtype=torch.float32, device=device)
Y = torch.as_tensor(Y, dtype=torch.float32, device=device)

# precompute cost matrix C (since X,Y fixed here)
with torch.no_grad():
    X_exp = X[:, None, :]
    Y_exp = Y[None, :, :]
    # Pairwise squared distances
    sq_dist = torch.sum((X_exp - Y_exp) ** 2, dim=-1).clamp(min=1e-12)
    # Compute cost explicitly in steps to avoid any precedence ambiguity:
    # raw_cost = -log(0.5 * ||x-y||^2); enforce non-negativity by clamping.
    log_arg = 0.5 * sq_dist  # >0 due to previous clamp
    C = -torch.log(log_arg)

for epoch in tqdm(range(epochs), desc="Training OT Model"):
    epsilon = epsilon_annealing(epoch)
    P = phi(X).squeeze(-1)  # (source_vertices,)
    S = psi(Y).squeeze(-1)  # (target_vertices,)

    P_pairwise_mat = P[:, None]  # (source_vertices, 1)
    S_pairwise_mat = S[None, :]  # (1, target_vertices)
    Z = torch.exp((P_pairwise_mat + S_pairwise_mat - C) / epsilon)

    penalty = epsilon * (Z.mean() - 1.0)
    loss_phi_term = -P.mean()
    loss_psi_term = -S.mean()
    L_entropy = penalty + loss_phi_term + loss_psi_term

    opt.zero_grad()
    L_entropy.backward()
    opt.step()

def compute_grad_phi_and_map(X, phi):
    """Return (phi(X), grad_phi(X), T(X)) using the full gradient formula.

    T = (1 - 2/(1 + |grad phi|^2)) * x + 2 grad phi / (1 + |grad phi|^2)
    """
    X_req = X.detach().requires_grad_(True)
    phi_vals = phi(X_req).squeeze(-1)
    (grad_phi,) = torch.autograd.grad(phi_vals.sum(), X_req, retain_graph=False)

    g_norm2 = grad_phi.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
    T = (1.0 - (2.0 / (1.0 + g_norm2))) * X_req + (2.0 / (1.0 + g_norm2)) * grad_phi
    return phi_vals.detach(), grad_phi.detach(), T.detach()

phi_vals_final, grad_phi_final, T_optical = compute_grad_phi_and_map(X, phi)

# visualize reflector as a 3D point cloud
with torch.no_grad():
    # compute the reflector points according to the phi function: exp(phi(sigma)) * X
    phi_vals = phi(X).squeeze(-1)
    #reflector_points = torch.exp(phi_vals)[:, None] * X
    reflector_points = torch.exp(phi_vals)[:, None] * X
    reflector_points_np = reflector_points.cpu().numpy()

    # sample additional points directly from X for evaluation
    # and compute their reflector images. No interpolation, just duplication for denser viz.
    N_EXTRA = 2723  # set 0 to disable
    if N_EXTRA > 0:
        extra_idx = torch.randint(0, X.shape[0], (N_EXTRA,), device=device)
        X_extra = X[extra_idx]
        phi_extra = phi(X_extra).squeeze(-1)
        reflector_extra = torch.exp(phi_extra)[:, None] * X_extra
        reflector_points_np = np.concatenate([
            reflector_points_np,
            reflector_extra.cpu().numpy()
        ], axis=0)

ps.init()
ps.set_ground_plane_mode("none")
ref = ps.register_point_cloud("reflector", reflector_points_np)
src = ps.register_point_cloud("source",    X.cpu().numpy())
tgt = ps.register_point_cloud("target",    Y.cpu().numpy())

for h in (ref, src, tgt):
    h.set_radius(0.005)
    h.set_point_render_mode("sphere")

src.set_color((0.10, 0.80, 0.10))  # green
tgt.set_color((0.90, 0.20, 0.20))  # red
ref.set_color((0.10, 0.60, 0.90))  # blue

with torch.no_grad():
    # Compute optical map points (already have T_optical) and nearest targets
    dists = torch.cdist(T_optical.to(device), Y)  # (N_src, N_tgt)
    nn_idx = torch.argmin(dists, dim=1)
    target_matched = Y[nn_idx]

# Use arrays we already have (reflector_points_np from earlier, and X)
source_points_np = X.cpu().numpy()
target_matched_np = target_matched.cpu().numpy()

base_N = source_points_np.shape[0]  # original source count
reflector_points_base_np = reflector_points_np[:base_N]
vertices_sr = np.concatenate([source_points_np, reflector_points_base_np], axis=0)
edges_sr = np.column_stack([np.arange(base_N), np.arange(base_N) + base_N])
ps.register_curve_network("edges source->reflector", vertices_sr, edges_sr)


R = reflector_points_np.shape[0]
vertices_rt = np.concatenate([reflector_points_np, target_matched_np], axis=0)
edges_rt = np.column_stack([np.arange(base_N), np.arange(base_N) + R])
ps.register_curve_network("edges reflector->target", vertices_rt, edges_rt)

ps.show()

print("--- Sanity checks ---")
print(f"C: min={C.min().item():.4f}, max={C.max().item():.4f}, mean={C.mean().item():.4f}")
print(f"C shape: {C.shape}")
print(f"Any NaN in C? {torch.isnan(C).any().item()} | Any Inf in C? {torch.isinf(C).any().item()}")
