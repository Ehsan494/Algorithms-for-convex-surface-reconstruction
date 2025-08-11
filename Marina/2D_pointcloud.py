import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
import trimesh
import gpytoolbox as gpy

from pathlib import Path

def sample_cloud(mesh_file, N, view='front'):
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
        vertices_2d = vertices[:, :2]  # x, y
    elif view == 'top':
        vertices_2d = vertices[:, [0, 2]]  # x, z
    elif view == 'side':
        vertices_2d = vertices[:, [2, 1]]  # y, z
    else:
        raise ValueError("view must be 'front', 'top', or 'side'")
    
    vertices_2d = vertices_2d.astype(np.float32)
    
    return vertices_2d, faces.astype(np.int32)

def solve_ot(X, Y):
    """
    Solve OT between two point sets X,Y ∈ ℝ^{N×2}, each carrying uniform mass 1/N.
    Returns:
      P_opt ∈ ℝ^{N×N} and optimal cost.
    """
    N = X.shape[0]
    a = np.ones(N) / N
    b = np.ones(N) / N

    # cost matrix: squared Euclidean distances
    C = cdist(X, Y, metric="sqeuclidean")

    # transport plan variable
    P = cp.Variable((N, N), nonneg=True)
    constraints = [
        cp.sum(P, axis=1) == a,   # each source i ships all its mass
        cp.sum(P, axis=0) == b    # each target j receives its mass
    ]
    objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
    prob = cp.Problem(objective, constraints)

    print("Solving OT problem...")
    cost = prob.solve()        # you can pass solver=cp.GUROBI if available
    print("Transport map ready.")
    
    return P.value, cost

def ui_callback():
    global s
    step_size = 0.05
    
    changed, s = psim.SliderFloat("Interpolation (s)", s, 0.0, 1.0)
    if changed:
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        
    if psim.Button("Prev"):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        
    psim.SameLine()
    if psim.Button("Next"):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        
    if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow,  repeat=True):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        
    if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=True):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)

s = 0.0 # interpolation parameter
N = 500 

# Choose orientation for projection: 'front', 'top', 'side'
source_vertices, source_faces = sample_cloud('data/bunny.obj', N, view='side')

target_vertices, target_faces = sample_cloud('data/spot.obj', N, view='front')

# Sample N points from each cloud
if len(source_vertices) >= N:
    source_idx = np.random.choice(len(source_vertices), N, replace=False)
    X = source_vertices[source_idx]
else:
    source_idx = np.random.choice(len(source_vertices), N, replace=True)
    X = source_vertices[source_idx]

if len(target_vertices) >= N:
    target_idx = np.random.choice(len(target_vertices), N, replace=False)
    Y = target_vertices[target_idx]
else:
    target_idx = np.random.choice(len(target_vertices), N, replace=True)
    Y = target_vertices[target_idx]

# Solve OT problem
P_opt, cost = solve_ot(X, Y)
print(f"OT cost: {cost:.4f}")

# Precompute barycentric images
a     = np.ones(N) / N
bary  = (P_opt @ Y) / a[:, None]   # shape (N,2)
bary  = bary.astype(np.float32)

ps.init()
ps.set_ground_plane_mode("none")  # Remove ground plane

# Add z=0 coordinate for 3D display of 2D points
X_3d = np.column_stack([X, np.zeros(N)])
cloud = ps.register_point_cloud("morph", X_3d)
cloud.set_radius(0.01)
cloud.set_point_render_mode("sphere")
cloud.set_color((0.1,0.6,0.9))

ps.set_user_callback(ui_callback)

print("Use the slider, arrow keys, or Prev/Next buttons to morph from shape 1 → shape 2.")
ps.show()
