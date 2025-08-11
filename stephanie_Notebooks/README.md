# Algorithms for Convex Surface Reconstruction

> **Mentor:** Prof. Nestor Guillén (Texas State University, US);
> **Project  Assistant:** Ehsan Shams (Alexandria University,EG)

Stephanie Atherton (Otis College of Art and Design, US); Minghao Ji (Columbia University, US); Amar KC (Howard University, US); Marina Oliveira Levay Reis (Minerva University, US);  Matthew Hellinger (Texas State University, US).     

**Project Abstract**: Reconstructing an unknown convex body **S** from the light-reflection pattern it casts onto a known screen can, under certain geometric assumptions, be reduced to a convex optimisation problem intimately linked to optimal transport and the Monge–Ampère equation.
This repository provides reproducible exploratory notebooks for the project *Algorithms for Convex Surface Reconstruction*.

## Preliminaries 

We will make use of [`cvxpy`](https://www.cvxpy.org/), a  modeling system for **convex optimization**. It lets you describe optimization problems almost exactly as you’d write them on paper, then automatically transforms and solves them using a solver.   

What can CVX do?

- Linear Programs (LPs)
- Quadratic Programs (QPs)
- Second-Order Cone Programs (SOCPs)
- Semidefinite Programs (SDPs)
- Norm minimization (ℓ₁, ℓ₂, ℓ∞) and other convex constructs

All of these follow the Disciplined Convex Programming (DCP) rules, ensuring automatic convexity verification.

What cannot CVX do?

- Non-convex problems (e.g., minimizing x³)
- Integer or mixed-integer programs
- Global non-convex solvers (branch-and-bound, genetic algorithms)

---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
