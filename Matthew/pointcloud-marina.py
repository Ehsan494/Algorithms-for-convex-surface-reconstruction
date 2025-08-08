import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
import trimesh
import gpytoolbox as gpy

from pathlib import Path

def sample_cloud(mesh_file, N):
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
    # Update mesh vertices
    mesh.vertices = vertices
    
    #points, _ = trimesh.sample.sample_surface(mesh, N)
    #print(f'points.shape: {points.shape}, vertices.shape: {vertices.shape}, faces.shape: {faces.shape}')

    #return points, vertices, faces
    
    # We only need the vertices (and faces for the mesh display)
    return vertices.astype(np.float32), faces.astype(np.int32)

def solve_ot(X, Y):
    """
    Solve OT between two point sets X,Y ∈ ℝ^{N×3}, each carrying uniform mass 1/N.
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
    global s, mesh_handle
    step_size = 0.05
    changed, s = psim.SliderFloat("Interpolation (s)", s, 0.0, 1.0)
    if changed:
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        verts = (1-s)*source_vertices + s*bary
        mesh_handle.update_vertex_positions(verts)
        
    if psim.Button("Prev"):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        verts = (1-s)*source_vertices + s*bary
        mesh_handle.update_vertex_positions(verts)
        
    psim.SameLine()
    if psim.Button("Next"):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        verts = (1-s)*source_vertices + s*bary
        mesh_handle.update_vertex_positions(verts)
        
    if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow,  repeat=True):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        verts = (1-s)*source_vertices + s*bary
        mesh_handle.update_vertex_positions(verts)
        
    if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=True):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        verts = (1-s)*source_vertices + s*bary
        mesh_handle.update_vertex_positions(verts)

s = 0.0     # interpolation parameter


# [rendundant part, will simplify] get vertices and faces from the source mesh
V, F = gpy.read_mesh('data/bunny.obj')
N = len(V)

#X, source_vertices, source_faces = sample_cloud('data/bunny.obj', N) # source points
#Y, _, _ = sample_cloud('data/spot.obj', N) # target points

# load bunny vertices & faces
source_vertices, source_faces = sample_cloud('data/bunny.obj', N)
X = source_vertices                # use the exact vertices as your source cloud

# load spot (cow) vertices & faces
target_vertices, target_faces = sample_cloud('data/spot.obj', N)
# if the cow has a different # of vertices, randomly pick N of them
if len(target_vertices) != N:
    idx = np.random.choice(len(target_vertices), N, replace=True)
    Y   = target_vertices[idx]
else:
    Y = target_vertices

# Solve OT problem
P_opt, cost = solve_ot(X, Y)
print(f"OT cost: {cost:.4f}")

# Precompute barycentric images
a     = np.ones(N) / N
bary  = (P_opt @ Y) / a[:, None]   # shape (N,3)
bary  = bary.astype(np.float32)

# Set up Polyscope & register morphing cloud and mesh
ps.init()

cloud = ps.register_point_cloud("morph", X)
cloud.set_radius(0.01)
cloud.set_point_render_mode("sphere")
cloud.set_color((0.1,0.6,0.9))

# Initial mesh from bunny OBJ
mesh_handle = ps.register_surface_mesh("morph_mesh", source_vertices, source_faces)

ps.set_user_callback(ui_callback)

print("Use the slider, arrow keys, or Prev/Next buttons to morph from shape 1 → shape 2 and see the reconstructed mesh.")
ps.show()
