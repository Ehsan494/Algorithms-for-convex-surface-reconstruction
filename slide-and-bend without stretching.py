import numpy as np
import matplotlib.pyplot as plt

n = 300
center_src = np.array([-2.0, 0.0])
radius = 1.0
r = np.sqrt(np.random.rand(n))*radius
theta = 2*np.pi*np.random.rand(n)
pts_src = center_src + np.column_stack((r*np.cos(theta), r*np.sin(theta)))

a = 1.3
b = 1/a
center_tgt = np.array([2.0, 0.0])

pts_rel = pts_src - center_src
pts_tgt = center_tgt + np.column_stack((a*pts_rel[:,0], b*pts_rel[:,1]))

fig, ax = plt.subplots(figsize=(8,4))
phi = np.linspace(0,2*np.pi,400)
ax.plot(center_src[0] + radius*np.cos(phi), center_src[1] + radius*np.sin(phi),
        linestyle='--', color='steelblue', label='Source Ω: circle')
ax.plot(center_tgt[0] + a*np.cos(phi), center_tgt[1] + b*np.sin(phi),
        color='orange', label="Target Ω′: same‑area ellipse")

idx = np.random.choice(n, 60, replace=False)
for i in idx:
    xs, ys = pts_src[i]
    xt, yt = pts_tgt[i]
    ax.arrow(xs, ys, xt-xs, yt-ys, head_width=0.05, head_length=0.1,
             length_includes_head=True, alpha=0.6, color='gray')

ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Slide‑and‑bend transformation (area preserved)')
ax.legend(loc='upper center', ncol=2)
plt.tight_layout()
python plt.savefig('circle_to_ellipse.png', dpi=300) 
