import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.animation import FuncAnimation,PillowWriter
def sunflower_circle_points(n, radius=1.0, center=(0.0, 0.0)):
    indices = np.arange(1, n + 1)
    phi = (np.sqrt(5) + 1) / 2
    theta = 2 * np.pi * indices / phi
    r = radius * np.sqrt(indices / n)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.stack((x, y), axis=-1)

# Create points
def equally_spaced_in_square(n_x, n_y, a=-5, b=-5, c=5, d=5):
    """
    Generate n_x x n_y grid of equally spaced points in square [a,c] x [b,d]
    """
    x_vals = np.linspace(a, c, n_x)
    y_vals = np.linspace(b, d, n_y)
    xv, yv = np.meshgrid(x_vals, y_vals)
    return np.stack((xv.ravel(), yv.ravel()), axis=-1)

x = sunflower_circle_points(100, radius=3.0)
y = equally_spaced_in_square(5,5)
# x= np.array([[-10,0],[10,0]])
# y= np.array([[-5,10],[50,10],[0,10]])

N = x.shape[0]
M = y.shape[0]

a = np.ones(N) / N
b = np.ones(M) / M

# Cost matrix
C = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

# Solve OT problem
P = cp.Variable((N, M))
constraints = [P >= 0, cp.sum(P, axis=1) == a, cp.sum(P, axis=0) == b]
objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
problem = cp.Problem(objective, constraints)
problem.solve()
P_val = P.value

# Plot source and target
plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1], color='blue', label='Source', s=10)
plt.scatter(y[:, 0], y[:, 1], color='red', label='Target', s=10)

# Draw arrows for each nonzero transport
original= []
mass = []
transported = []

threshold = 1e-5
for i in range(N):
    for j in range(M):
        if P_val[i, j] > threshold:
            # Color can indicate amount of mass
            original.append(x[i])
            transported.append(y[j])
            mass.append(P_val[i, j])
            alpha = min(1.0, P_val[i, j] * N * 5)
            plt.arrow(x[i, 0], x[i, 1],
                      y[j, 0] - x[i, 0], y[j, 1] - x[i, 1],
                      color=(0, 0, 0, alpha),
                      width=0.0005, head_width=0.02, length_includes_head=True)
original = np.array(original)
transported = np.array(transported)
mass = np.array(mass)

# Animation setup
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(original[:, 0], original[:, 1], color='blue', s=10, label='Interpolated')

# Static points
ax.scatter(y[:, 0], y[:, 1], color='red', s=10, label='Targets')
#ax.scatter(original[:, 0], original[:, 1], color='gray', alpha=0.3, s=10, label='Source')
ax.scatter(transported[:, 0], transported[:, 1], color='green', alpha=0.3, s=10, label='T(x)')

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.set_title("Morphing Source Points to Transport Targets")
ax.grid(True)
ax.legend()

# Update function
def update(frame):
    alpha = frame / 100
    interp = (1 - alpha) * original + alpha * transported
    sc.set_offsets(interp)
    return sc,

# Animate
anim = FuncAnimation(fig, update, frames=101, interval=50, blit=True)
anim.save("coupling.gif", writer=PillowWriter(fps=30))
plt.show()

