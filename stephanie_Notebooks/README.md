# Algorithms for Convex Surface Reconstruction

> **Mentor:** [Prof. Nestor Guillén](https://www.ndguillen.com/) (Texas State University, US);
> **Project  Assistant:** Ehsan Shams (Alexandria University,EG)

Stephanie Atherton (Otis College of Art and Design, US); Minghao Ji (Columbia University, US); Amar KC (Howard University, US); Marina Oliveira Levay Reis (Minerva University, US);  Matthew Hellinger (Texas State University, US).     

**Project Abstract**: Reconstructing an unknown [convex body](https://en.wikipedia.org/wiki/Convex_body) **S** from the light-reflection pattern it casts onto a known screen can, under certain [geometric assumptions](https://github.com/Ehsan494/Algorithms-for-convex-surface-reconstruction/blob/main/Selected%20Papers/2019_PrimerOnGJE.pdf), be reduced to a [convex optimization](https://en.wikipedia.org/wiki/Convex_optimization)  problem intimately linked to [optimal transport](https://github.com/Ehsan494/Algorithms-for-convex-surface-reconstruction/blob/main/Selected%20Papers/Solomon_OT_DiscreteDomains.pdf) and the [Monge–Ampère equation](https://en.wikipedia.org/wiki/Monge%E2%80%93Amp%C3%A8re_equation).
This repository provides reproducible exploratory notebooks for the project *Algorithms for Convex Surface Reconstruction*.

## Required Packages
We recommend setting up a python environment with the required packages:
- `cvxpy`: convex optimization 
- `trimesh`: loading, manipulating, visualizing triangular meshes
- `triangle`: 2D triangular mesh generator 
- `gpytoolbox`: geometry processing utility functions
- `polyscope`: 3D visualization
- `svgpathtools`: geometry analysis of SVG files
- `pathlib`: handles file system paths 
- `torch`: deep learning
- `tqdm`: progress bars for loops
- `scipy`: scientific computing 
- `numpy`: numerical computing
- `matplotlib`: plotting and visualization
  
## Project Outline 

```
Algorithms-for-conves-surface-reconstruction/
├── OT_cvx/
|   ├── ot_cvx.py               
│   └── README.md                  
├── 1D_example/
│   ├── 1D_example.ipynb              
│   └── README.md
├── 2D_ShapeInterpolation/
├── data/  
├── svg_pointcloud.py              
│   └── README.md     
├── 3D_ShapeInterpolation/
│   ├── data/
|   ├── pointcloud_marina.ipynb
|   ├── pointcloud.ipynb
|   └── README.md
├── FarFieldReflectorProblem/
    ├── CVX/
    ├── Torch/
    └── README.md
 
```




---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
