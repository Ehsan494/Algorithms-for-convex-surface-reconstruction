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
import triangle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def circle_mesh_triangle(radius=0.5, n_boundary=128, center=(0.0, 0.0),
                         target_edge_len=None, max_area=None, quality=30, z=0.0):
    """
    Triangulate a filled circle (disk) with Triangle and return (V3, F, markers).

    Returns:
      V3: (Nv,3) float32 vertices lying on z=const plane
      F:  (Nt,3) int32  triangle indices
      markers: (Nv,) int; 1 on boundary vertices (from PSLG), 0 interior
    """
    # boundary polygon (PSLG)
    theta = np.linspace(0.0, 2*np.pi, n_boundary, endpoint=False)
    poly = np.c_[center[0] + radius*np.cos(theta),
                 center[1] + radius*np.sin(theta)]
    segs = np.c_[np.arange(n_boundary), (np.arange(n_boundary)+1) % n_boundary]

    A = {"vertices": poly.astype(np.float64), "segments": segs.astype(np.int32)}

    # Triangle options
    # p: PSLG input, qXX: min angle, aA: max area, Q: quiet
    if max_area is None and target_edge_len is not None:
        # equilateral area ≈ (sqrt(3)/4)*h^2
        max_area = (np.sqrt(3.0)/4.0) * (float(target_edge_len)**2)
    opts = f"pQq{int(quality)}" + (f"a{max_area:.8f}" if max_area is not None else "")

    B = triangle.triangulate(A, opts)

    V2 = B["vertices"].astype(np.float64)          # (Nv,2)
    F  = B["triangles"].astype(np.int32)           # (Nt,3)
    markers = B.get("vertex_markers", np.zeros(len(V2), dtype=np.int32))

    # Lift to z-plane for 3D pipeline
    V3 = np.column_stack([V2, np.full((V2.shape[0],), z, dtype=V2.dtype)]).astype(np.float32)
    return V3, F, markers

def sample_points_from_mesh(V, F, N, rng=np.random):
    """
    Uniformly sample N points from a triangle mesh (V:(Nv,3), F:(Nt,3)).
    Returns (N,3) float32.
    """
    # triangle areas
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    areas = 0.5*np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    probs = areas / areas.sum()

    tri_idx = rng.choice(len(F), size=N, p=probs)
    v0, v1, v2 = V[F[tri_idx,0]], V[F[tri_idx,1]], V[F[tri_idx,2]]

    # barycentric sampling
    r1 = np.sqrt(rng.rand(N)).astype(np.float64)
    r2 = rng.rand(N).astype(np.float64)
    a = 1.0 - r1
    b = r1*(1.0 - r2)
    c = r1*r2
    pts = (a[:,None]*v0 + b[:,None]*v1 + c[:,None]*v2).astype(np.float32)
    return pts

# sample point clouds 
def sample_cloud(mesh_file, view='front'):
    """
    Load a 3D mesh, normalize into [-1,1]^3, and project to a flattened 3D plane.

    We keep 3 coordinates (Nx3) so downstream models operate in 3D, but the
    geometry lies on a coordinate plane (z=0, y=0, or x=0 depending on view).

    Args:
        mesh_file: Path to mesh file (relative to this script or absolute)
        view: 'front' (xy plane, z=0), 'top' (xz plane, y=0), or 'side' (yz plane, x=0)
    Returns:
        (points_3d_flat, faces)
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

    if view == 'front':
        # (x,y,0)
        vertices_out = np.stack([vertices[:,0], vertices[:,1], np.zeros_like(vertices[:,0])], axis=1)
    elif view == 'top':
        # (x,0,z)
        vertices_out = np.stack([vertices[:,0], np.zeros_like(vertices[:,0]), vertices[:,2]], axis=1)
    elif view == 'side':
        # (0,y,z)
        vertices_out = np.stack([np.zeros_like(vertices[:,0]), vertices[:,1], vertices[:,2]], axis=1)
    else:
        raise ValueError("view must be 'front', 'top', or 'side'")

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
N = 500  # number of points to sample from each point cloud

# define the source and target potential functions
phi = Phi(Ds).to(device)
psi = Psi(Dt).to(device)
opt = torch.optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=1e-3)

epochs = 1000
# the epsilon parameter controls the entropy regularization strength
# larger epsilon means more entropy regularization; smaller epsilon means less regularization (transport concentrates toward the true OT plan)
epsilon = 0.001

# choose orientations for 2D projection
source_vertices, source_faces = sample_cloud('data/bunny.obj', view='front')
target_vertices, target_faces = sample_cloud('data/spot.obj', view='side')

'''
# sample points from each point cloud
#X = source_vertices
X = source_vertices + np.ones((502,3))
#Y = target_vertices
#Y = target_vertices - 2*np.ones((3225,3))
Y = np.array([[5,10,-5]])

# normalize X and Y to unit vectors
X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
Y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-12)
'''
# ---- build source from Triangle circle ----
V_src, F_src, markers_src = circle_mesh_triangle(radius=0.5, n_boundary=128, target_edge_len=0.06)

# pick: filled disk (sampled) OR boundary ring
# A) disk:
X_np = sample_points_from_mesh(V_src, F_src, N)
# B) ring (comment A, uncomment B if you want perimeter):
# X_np = V_src[markers_src.astype(bool)]
# if len(X_np) > N: X_np = X_np[np.random.choice(len(X_np), N, replace=False)]

# ---- target (your existing) ----
target_vertices, target_faces = sample_cloud('data/spot.obj', view='side')
Y_np = target_vertices   # or your existing Y

# Convert to torch tensors
X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
Y = torch.as_tensor(Y_np, dtype=torch.float32, device=device)

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

    # penalization term 
    Z = torch.exp((P + S - C)/epsilon)

    L_entropy = epsilon * (Z.mean() - 1.0) - P.mean() - S.mean()

    opt.zero_grad()
    L_entropy.backward()
    opt.step()

# visualize reflector as a 3D point cloud
with torch.no_grad():
    # compute the reflector points according to the phi function: exp(phi(sigma)) * X
    phi_vals = phi(X).squeeze(-1)
    #reflector_points = torch.exp(phi_vals)[:, None] * X
    reflector_points = torch.exp(phi_vals+0.5)[:, None] * X
    reflector_points_np = reflector_points.cpu().numpy()

# visualize reflector as a 3D point cloud
with torch.no_grad():
    # compute the reflector points according to the phi function: exp(phi(sigma)) * X
    source_points = X
    source_points_np = source_points.cpu().numpy()

ps.init()
ps.set_ground_plane_mode("none")
#ps_cloud = ps.register_point_cloud("reflector 3D points", reflector_points_np)
#print("Reflector 3D point cloud visualized in Polyscope.")
#ps_cloud = ps.register_point_cloud("source points", source_points_np)
#print("Source point cloud visualized in Polyscope.")
#ps_cloud = ps.register_point_cloud("target points", Y.cpu().numpy())
#print("Target point cloud visualized in Polyscope.")
#ps_cloud.set_radius(0.005)
#ps_cloud.set_point_render_mode("sphere")
#ps_cloud.set_color((0.9, 0.6, 0.1))
ref = ps.register_point_cloud("reflector", reflector_points_np)
src = ps.register_point_cloud("source",    source_points_np)
tgt = ps.register_point_cloud("target",    Y.cpu().numpy())

for h in (ref, src, tgt):
    h.set_radius(0.005)
    h.set_point_render_mode("sphere")

src.set_color((0.10, 0.80, 0.10))  # green
tgt.set_color((0.90, 0.20, 0.20))  # red
ref.set_color((0.10, 0.60, 0.90))  # blue
ps.show()

print("--- Sanity checks ---")
print(f"C: min={C.min().item():.4f}, max={C.max().item():.4f}, mean={C.mean().item():.4f}")
print(f"C shape: {C.shape}")
print(f"Any NaN in C? {torch.isnan(C).any().item()} | Any Inf in C? {torch.isinf(C).any().item()}")

