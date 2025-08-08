import numpy as np
import polyscope as ps
import gpytoolbox as gpy


# Load your mesh
V, F = gpy.read_mesh("spot.obj")

def random_points_on_3d_mesh(V, F, n_points):
    # V: (N, 3) vertices
    # F: (M, 3) triangle indices

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Compute area using cross product in 3D
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    prob = areas / areas.sum()

    # Sample triangles proportional to area
    tri_idx = np.random.choice(len(F), size=n_points, p=prob)
    tri = F[tri_idx]

    # Barycentric coordinates
    u = np.random.rand(n_points, 1)
    v = np.random.rand(n_points, 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v

    p0 = V[tri[:, 0]]
    p1 = V[tri[:, 1]]
    p2 = V[tri[:, 2]]

    points = u * p0 + v * p1 + w * p2
    return points
points = random_points_on_3d_mesh(V, F, 500)
# Save the point cloud as an OBJ file
def save_point_cloud_as_obj(filename, points):
    with open(filename, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
#save_point_cloud_as_obj("500_spot.obj", points)
ps.init()
ps.register_point_cloud("Sampled Points", points, radius=0.002)
ps.show()