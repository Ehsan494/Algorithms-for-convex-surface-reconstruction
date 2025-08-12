# OT with Convex Optimization Demo 

This Jupyter notebook demonstrates how to solve and visualize a simple [optimal transport (OT)](https://github.com/Ehsan494/Algorithms-for-convex-surface-reconstruction/blob/main/Selected%20Papers/Solomon_OT_DiscreteDomains.pdf) problem using Python, specifically with the `cvxpy` library for convex optimization and `matplotlib` for plotting.

We define two discrete probability distributions: a "source" (a) and a "target" (b), each over three locations/points on the real line (x and y) and construct a cost matrix C where each entry is the squared distance between a source and a target location.  We then set  up a convex optimization problem to find the transport plan P (a matrix), representing how much mass to move from each source to each target and  add  constraints so that the row sums of P equal the source distribution and the column sums equal the target distribution (ensuring all mass is transported correctly).  The objective is to minimize the total cost of transport, calculated as the sum of each transported mass times its cost.  We solve the convex optimization problem using `cvxpy` to find the optimal transport plan and the minimal total cost.

## Visualiation 
The  notebook provides a clear, visual, and practical example of how to compute and interpret the optimal transport plan, printing the optimal transport cost and the transport plan matrix and plotting via `matplotlib`.  We have:
 
1. The source and target distributions as bar plots. 
2. A flow diagram showing how mass is transported from each source to each target. Here, the lines connect source to target, with thickness proportional to the amount of mass moved and the value of the transported mass is labeled on each line. The source and target nodes are shown as circles sized by their probability mass

---


## License <a id="license"></a>

MIT © 2025 **Nestor Guillén & contributors**.
See [`LICENSE`](LICENSE) for full terms.
