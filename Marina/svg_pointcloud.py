import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import cvxpy as cp
from scipy.spatial.distance import cdist
from svgpathtools import svg2paths
from pathlib import Path

def sample_svg(svg_file, N):
    """
    Sample points from SVG paths.
    
    Args:
        svg_file: Path to SVG file
        N: Number of points to sample
    """
    # Resolve svg_file to an absolute path relative to this script
    svg_path = Path(svg_file)
    if not svg_path.is_absolute():
        svg_path = Path(__file__).parent / svg_file
        
    paths, _ = svg2paths(str(svg_path))
    
    # compute total length
    lengths = np.array([p.length() for p in paths], dtype=float)
    if lengths.sum() == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = []
    for p, L in zip(paths, lengths):
        k = max(1, int(N * (L / lengths.sum())))
        # sample uniformly by arclength using parameter t in [0,1]
        ts = np.linspace(0, 1, k, endpoint=False)
        for t in ts:
            z = p.point(t)        # complex
            pts.append([z.real, z.imag])
    
    P = np.asarray(pts, dtype=np.float32)
    
    if len(P) == 0:
        return np.empty((0, 2), dtype=np.float32)

    # normalize to [-1,1]^2, center at origin, flip Y (SVG y-down -> y-up)
    P[:, 1] *= -1.0
    P -= P.mean(axis=0)
    P /= np.abs(P).max()
    
    return P

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

# Global variables for UI
s = 0.0
X = None
bary = None
cloud = None

def ui_callback():
    global s, cloud, X, bary
    step_size = 0.05
    
    changed, s = psim.SliderFloat("Interpolation (s)", s, 0.0, 1.0)
    if changed:
        Xs = (1-s)*X + s*bary
        # Add z=0 coordinate for 3D display
        Xs_3d = np.column_stack([Xs, np.zeros(len(Xs))])
        cloud.update_point_positions(Xs_3d)
        
    if psim.Button("Prev"):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        Xs_3d = np.column_stack([Xs, np.zeros(len(Xs))])
        cloud.update_point_positions(Xs_3d)
        
    psim.SameLine()
    if psim.Button("Next"):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        Xs_3d = np.column_stack([Xs, np.zeros(len(Xs))])
        cloud.update_point_positions(Xs_3d)
        
    if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow, repeat=True):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        Xs_3d = np.column_stack([Xs, np.zeros(len(Xs))])
        cloud.update_point_positions(Xs_3d)
        
    if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=True):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        Xs_3d = np.column_stack([Xs, np.zeros(len(Xs))])
        cloud.update_point_positions(Xs_3d)

# Configuration
N = 500  # Number of points to sample from each SVG

# Load SVG shapes
print("Loading SVG shapes...")
source_points = sample_svg('Rase_Gingerbread_Man_1.svg', N)
target_points = sample_svg('Rase_Gingerbread_Man_2.svg', N)

# Ensure both have the same number of points
min_points = min(len(source_points), len(target_points))
if min_points == 0:
    print("Error: One or both SVG files could not be loaded or have no paths.")
    exit()
    
if len(source_points) > min_points:
    idx = np.random.choice(len(source_points), min_points, replace=False)
    source_points = source_points[idx]
if len(target_points) > min_points:
    idx = np.random.choice(len(target_points), min_points, replace=False)
    target_points = target_points[idx]

X = source_points
Y = target_points

print(f"Source shape: {len(X)} points")
print(f"Target shape: {len(Y)} points")

# Solve OT problem
P_opt, cost = solve_ot(X, Y)
print(f"OT cost: {cost:.4f}")

# Precompute barycentric images
N = len(X)
a = np.ones(N) / N
bary = (P_opt @ Y) / a[:, None]   # shape (N,2)
bary = bary.astype(np.float32)

# Set up Polyscope
ps.init()
ps.set_ground_plane_mode("none")  # Remove ground plane

# Add z=0 coordinate for 3D display of 2D points
X_3d = np.column_stack([X, np.zeros(N)])
cloud = ps.register_point_cloud("morph", X_3d)
cloud.set_radius(0.004)
cloud.set_point_render_mode("quad")
cloud.set_color((0.1, 0.6, 0.9))

ps.set_user_callback(ui_callback)

print("Use the slider, arrow keys, or Prev/Next buttons to morph between SVG shapes.")
ps.show()
