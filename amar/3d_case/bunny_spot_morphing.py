import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.animation import FuncAnimation
import gpytoolbox as gpy
import trimesh
import polyscope as ps
import polyscope.imgui as psim
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots


mesh1 = trimesh.load("1000_spot.obj")
x = mesh1.vertices
mesh2 = trimesh.load("1000_bunny.obj")
y = mesh2.vertices

N = x.shape[0]
M = y.shape[0]
a = np.ones(N) / N
b = np.ones(M) / M

# Cost matrix (squared distances)
C = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

# Optimal transport
P = cp.Variable((N, M))
constraints = [P >= 0, cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
cp.Problem(objective, constraints).solve()
P_val = P.value

# Extract mass transport pairs
original, transported, mass = [], [], []
for i in range(N):
    for j in range(M):
        if P_val[i, j] > 1e-5:
            original.append(x[i])
            transported.append(y[j])
            mass.append(P_val[i, j])

original = np.array(original)
transported = np.array(transported)
mass = np.array(mass)

# Initialize Polyscope
ps.init()
ps.set_ground_plane_mode("none")

# Register source, target, and interpolated points
#ps.register_point_cloud("source (gray)", original, radius=0.003).set_color((0.5, 0.5, 0.5))
#ps.register_point_cloud("target (green)", transported, radius=0.003).set_color((0.0, 1.0, 0.0))
interpolated_pc = ps.register_point_cloud("interpolated (blue)", original.copy(), radius=0.003)
interpolated_pc.set_color((0.0, 0.3, 1.0))

# Slider-controlled animation
alpha_val = [0.0]
def callback():
    changed, alpha = psim.SliderFloat("Interpolation α", alpha_val[0], 0.0, 1.0)
    if changed:
        alpha_val[0] = alpha
        interp = (1 - alpha) * original + alpha * transported
        interpolated_pc.update_point_positions(interp)

ps.set_user_callback(callback)
ps.show()