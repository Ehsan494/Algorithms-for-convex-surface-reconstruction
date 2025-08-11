import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import cdist
import trimesh

def sample_cloud(mesh_file, N):
    mesh = trimesh.load(mesh_file, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, N)
    return points

def generate_point_cloud(shape, n, surface_only=False, **kwargs):
    """
    Generate n points in or on a given shape, centered at the origin.

    Parameters
    ----------
    shape : str
        One of 'cube', 'sphere', 'cylinder', 'cone',
        'pyramid', 'rectangular_prism', 'ellipsoid'.
    n : int
        Number of points.
    surface_only : bool
        If True, sample only on the surface; else sample in the volume.
    **kwargs :
        size parameters for each shape:
        - cube:        size (edge length; default=1)
        - sphere:      radius (default=0.5)
        - cylinder:    radius (default=0.5), height (default=1)
        - cone:        radius (base; default=0.5), height (default=1)
        - pyramid:     base (edge length; default=1), height (default=1)
        - rectangular_prism: dims (tuple w,h,d; default=(1,1,1))
        - ellipsoid:   radii (tuple rx,ry,rz; default=(0.5,0.5,0.5))
    """
    if shape == 'cube':
        size = kwargs.get('size', 1.0)
        half = size/2
        if not surface_only:
            return (np.random.rand(n,3)-0.5)*size
        # ------ surface sampling on 6 faces ------
        faces = np.random.randint(0, 6, n)
        pts = np.empty((n,3))
        # face 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z
        for f in range(6):
            m = (faces == f)
            cnt = m.sum()
            if cnt == 0: continue
            uv = (np.random.rand(cnt,2)-0.5)*size
            if f == 0: pts[m] = np.column_stack([ half*np.ones(cnt), uv])
            if f == 1: pts[m] = np.column_stack([-half*np.ones(cnt), uv])
            if f == 2: pts[m] = np.column_stack([uv[:,0],  half*np.ones(cnt), uv[:,1]])
            if f == 3: pts[m] = np.column_stack([uv[:,0], -half*np.ones(cnt), uv[:,1]])
            if f == 4: pts[m] = np.column_stack([uv,  half*np.ones(cnt)])
            if f == 5: pts[m] = np.column_stack([uv, -half*np.ones(cnt)])
        return pts

    elif shape == 'sphere':
        r = kwargs.get('radius', 0.5)
        # volume
        if not surface_only:
            u = np.random.rand(n)**(1/3)
            v = np.random.randn(n,3)
            v /= np.linalg.norm(v, axis=1)[:,None]
            return v * u[:,None] * r
        # surface
        v = np.random.randn(n,3)
        v /= np.linalg.norm(v, axis=1)[:,None]
        return v * r

    elif shape == 'cylinder':
        r = kwargs.get('radius', 0.5)
        h = kwargs.get('height', 1.0)
        half_h = h/2
        if not surface_only:
            radii = np.sqrt(np.random.rand(n)) * r
            theta = 2*np.pi*np.random.rand(n)
            x = radii*np.cos(theta)
            y = radii*np.sin(theta)
            z = np.random.rand(n)*h - half_h
            return np.vstack([x,y,z]).T
        # lateral vs. top vs. bottom by area
        A_lat = 2*np.pi*r*h
        A_cap = np.pi*r*r
        p = [A_lat, A_cap, A_cap]
        p = np.array(p)/sum(p)
        choice = np.random.choice(3, n, p=p)
        pts = np.empty((n,3))
        # lateral
        m = choice==0
        cnt = m.sum()
        theta = 2*np.pi*np.random.rand(cnt)
        z = np.random.rand(cnt)*h - half_h
        pts[m] = np.column_stack([r*np.cos(theta), r*np.sin(theta), z])
        # top cap
        m = choice==1
        cnt = m.sum()
        rad = np.sqrt(np.random.rand(cnt))*r
        theta = 2*np.pi*np.random.rand(cnt)
        pts[m] = np.column_stack([rad*np.cos(theta), rad*np.sin(theta),  half_h])
        # bottom cap
        m = choice==2
        cnt = m.sum()
        rad = np.sqrt(np.random.rand(cnt))*r
        theta = 2*np.pi*np.random.rand(cnt)
        pts[m] = np.column_stack([rad*np.cos(theta), rad*np.sin(theta), -half_h])
        return pts

    elif shape == 'cone':
        R = kwargs.get('radius', 0.5)
        H = kwargs.get('height', 1.0)
        half_H = H/2
        L = np.sqrt(H*H + R*R)
        if not surface_only:
            u = np.random.rand(n)
            z = H * (1 - (1-u)**(1/3))
            max_r = R*(1 - z/H)
            rad = np.sqrt(np.random.rand(n))*max_r
            theta = 2*np.pi*np.random.rand(n)
            pts = np.column_stack([rad*np.cos(theta),
                                   rad*np.sin(theta),
                                   z - half_H])
            return pts
        # lateral vs. base by area
        A_lat  = np.pi*R*L
        A_base = np.pi*R*R
        probs = np.array([A_lat, A_base])/(A_lat + A_base)
        choice = np.random.choice(2, n, p=probs)
        pts = np.empty((n,3))
        # lateral
        m = choice==0
        cnt = m.sum()
        u = np.random.rand(cnt)
        l = L * (1 - np.sqrt(1-u))
        zloc = l*H/L - half_H
        rloc = R*(1 - l/L)
        theta = 2*np.pi*np.random.rand(cnt)
        pts[m] = np.column_stack([rloc*np.cos(theta),
                                  rloc*np.sin(theta),
                                  zloc])
        # base
        m = choice==1
        cnt = m.sum()
        rad = np.sqrt(np.random.rand(cnt))*R
        theta = 2*np.pi*np.random.rand(cnt)
        pts[m] = np.column_stack([rad*np.cos(theta),
                                  rad*np.sin(theta),
                                  -half_H])
        return pts

    elif shape == 'pyramid':
        B = kwargs.get('base', 1.0)
        H = kwargs.get('height', 1.0)
        half_B, half_H = B/2, H/2
        slant = np.sqrt(half_B*half_B + H*H)
        if not surface_only:
            u = np.random.rand(n)
            z = H*(1 - (1-u)**(1/3))
            half_s = half_B*(1 - z/H)
            x = (np.random.rand(n)*2 -1)*half_s
            y = (np.random.rand(n)*2 -1)*half_s
            return np.column_stack([x,y,z-half_H])
        # lateral vs. base by area
        A_lat  = 2*B*slant
        A_base = B*B
        probs = np.array([A_lat, A_base])/(A_lat + A_base)
        choice = np.random.choice(2, n, p=probs)
        pts = np.empty((n,3))
        # lateral faces
        m = choice==0
        cnt = m.sum()
        faces = np.random.randint(0,4, cnt)
        # sample barycentrically on each triangular face
        r1 = np.random.rand(cnt);  r2 = np.random.rand(cnt)
        over = (r1+r2)>1; r1[over], r2[over] = 1-r1[over], 1-r2[over]
        apex = np.array([0,0, half_H])
        base_c = np.array([[ half_B,  half_B,-half_H],
                           [-half_B,  half_B,-half_H],
                           [-half_B, -half_B,-half_H],
                           [ half_B, -half_B,-half_H]])
        V0 = base_c[faces]
        V1 = base_c[(faces+1)%4]
        pts[m] = apex + (V0-apex)*r1[:,None] + (V1-apex)*r2[:,None]
        # base
        m = choice==1
        cnt = m.sum()
        x = (np.random.rand(cnt)*2 -1)*half_B
        y = (np.random.rand(cnt)*2 -1)*half_B
        pts[m] = np.column_stack([x,y, -half_H])
        return pts

    elif shape == 'rectangular_prism':
        w,d,h = kwargs.get('dims',(1.0,1.0,1.0))
        half = np.array([w/2,d/2,h/2])
        if not surface_only:
            x = (np.random.rand(n)-0.5)*w
            y = (np.random.rand(n)-0.5)*d
            z = (np.random.rand(n)-0.5)*h
            return np.vstack([x,y,z]).T
        # sample faces by area
        areas = np.array([d*h, d*h, w*h, w*h, w*d, w*d])
        probs = areas/areas.sum()
        faces = np.random.choice(6, n, p=probs)
        pts = np.zeros((n,3))
        for i in range(6):
            m = faces==i; cnt = m.sum()
            if cnt==0: continue
            uv = (np.random.rand(cnt,2)-0.5)*[d,h] if i<2 else \
                 (np.random.rand(cnt,2)-0.5)*([w,h] if i<4 else [w,d])
            if i==0: pts[m] = np.column_stack([ half[0], uv])
            if i==1: pts[m] = np.column_stack([-half[0], uv])
            if i==2: pts[m] = np.column_stack([uv[:,0],  half[1], uv[:,1]])
            if i==3: pts[m] = np.column_stack([uv[:,0], -half[1], uv[:,1]])
            if i==4: pts[m] = np.column_stack([uv,  half[2]])
            if i==5: pts[m] = np.column_stack([uv, -half[2]])
        return pts

    elif shape == 'ellipsoid':
        rx,ry,rz = kwargs.get('radii',(0.5,0.5,0.5))
        if not surface_only:
            u = np.random.rand(n)**(1/3)
            v = np.random.randn(n,3)
            v /= np.linalg.norm(v,axis=1)[:,None]
            return v*u[:,None]*np.array([rx,ry,rz])
        # sample direction, then project to ellipsoid
        u = np.random.randn(n,3)
        u /= np.linalg.norm(u,axis=1)[:,None]
        denom = np.sqrt((u[:,0]/rx)**2 + (u[:,1]/ry)**2 + (u[:,2]/rz)**2)
        return (u / denom[:,None])

    else:
        raise ValueError(f"Unknown shape '{shape}'")

def flatten_cloud(pts, axis):
    """
    Orthographically flatten a 3D point cloud onto a coordinate plane
    by zeroing the chosen axis. Keeps shape (n,3).

    axis: 'x' -> onto YZ plane (vertical)
          'y' -> onto XZ plane (vertical)
          'z' -> onto XY plane (horizontal)
    """
    idx = {'x':0, 'y':1, 'z':2}[axis.lower()]
    out = np.array(pts, copy=True)
    out[:, idx] = 0.0
    return out

def project_cloud(pts, view='front'):
    """
    Return a 2D array by selecting two axes for plotting.

    view: 'front' -> (x,y)
          'top'   -> (x,z)
          'side'  -> (y,z)
          or pass ('x','z'), (0,2), etc.
    """
    name2idx = {'x':0,'y':1,'z':2}
    presets  = {'front':(0,1), 'top':(0,2), 'side':(1,2)}
    if isinstance(view, str):
        i, j = presets[view]
    else:
        a, b = view
        i = name2idx[a] if isinstance(a, str) else int(a)
        j = name2idx[b] if isinstance(b, str) else int(b)
    return pts[:, (i, j)]

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

    cost = prob.solve()        # you can pass solver=cp.GUROBI if available
    return P.value, cost

def ui_callback():
    global s
    step_size = 0.05
    # Slider for s in [0,1]
    changed, s = psim.SliderFloat("Interpolation (s)", s, 0.0, 1.0)
    if changed:
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)

    # Optional Prev/Next buttons for discrete stepping
    if psim.Button("Prev"):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
    psim.SameLine()
    if psim.Button("Next"):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
        
    # ← / → keys with repeat
    if psim.IsKeyPressed(psim.ImGuiKey_LeftArrow,  repeat=True):
        s = max(0.0, s - step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)
    if psim.IsKeyPressed(psim.ImGuiKey_RightArrow, repeat=True):
        s = min(1.0, s + step_size)
        Xs = (1-s)*X + s*bary
        cloud.update_point_positions(Xs)

s = 0.0     # interpolation parameter

# Generate point clouds with same number of points
N = 500
#X = generate_point_cloud('cube',  N, surface_only=True, size=1.0).astype(np.float32)
#Y = generate_point_cloud('sphere',N, surface_only=True, radius=0.5).astype(np.float32)
#X = sample_cloud('bunny.obj', N)   # source points
#Y = sample_cloud('spot.obj',    N)   # target points
#w = v = np.full(N, 1.0/N)           # equal weights

# Generate points ON the sphere (3D)
sphere_pts = generate_point_cloud('sphere', N, surface_only=True, radius=0.5).astype(np.float32)

# Horizontal circle (XY plane): zero Z
horizontal_circle_3d = flatten_cloud(sphere_pts, 'z')      # lies in XY

# Vertical circle (XZ plane): zero Y  [or use 'x' for YZ]
vertical_circle_xz_3d = flatten_cloud(sphere_pts, 'y')     # lies in XZ

# Use the 3D arrays directly; do NOT call project_cloud here
X = horizontal_circle_3d
Y = vertical_circle_xz_3d

# Solve OT problem
P_opt, cost = solve_ot(X, Y)
print(f"OT cost: {cost:.4f}")

# Precompute barycentric images
a     = np.ones(N) / N
bary  = (P_opt @ Y) / a[:, None]   # shape (N,3)
bary  = bary.astype(np.float32)

# Set up Polyscope & register a single morphing cloud
ps.init()
cloud = ps.register_point_cloud("morph", X)
cloud.set_radius(0.01)
cloud.set_point_render_mode("sphere")
cloud.set_color((0.1,0.6,0.9))

ps.set_user_callback(ui_callback)

print("Use the slider, arrow keys, or Prev/Next buttons to morph from shape 1 → shape 2.")
ps.show()
