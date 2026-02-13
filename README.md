# pppca

**Point Process Principal Component Analysis (PPPCA)**

`pppca` implements Gram PCA for multivariate point processes on \([0,1]^d\). 
Instead of treating data as fixed-length vectors, this library handles **point processes**—sets of points in space where the number of points can vary between observations. It computes a centered Gram matrix, performs eigendecomposition, and provides tools to evaluate the learned "eigenfunctions" (principal components) at any location in the domain.

This is particularly useful for dimensionality reduction and exploratory analysis of:
- Spike trains (1D)
- Spatial point patterns (2D/3D)
- Event logs or sparse observational data

## Basic Usage

### Input Format
The main function `pppca` expects a list of tensors.
- **Input**: A python `list` of `n` observations.
- **Observation**: A `torch.Tensor` of shape `(k_i, d)`, where `k_i` is the number of points in that specific observation and `d` is the dimension (e.g., `d=2` for 2D coordinates).
- **Domain**: All coordinates should be normalized to `[0, 1]`.

### Quickstart

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pppca.core import pppca

# 1. Generate synthetic data (e.g., 25 observations in 2D)
# Each element in 'processes' is a tensor of shape (num_points, 2)
d = 2
n_obs = 25
processes = []
torch.manual_seed(42)

for _ in range(n_obs):
    # Create random points in ^2
    num_points = torch.randint(low=5, high=20, size=(1,)).item()
    points = torch.rand((num_points, d))
    processes.append(points)

# 2. Run PPPCA
results = pppca(processes, Jmax=2)

# 3. Inspect Results
print("Eigenvalues:", results["eigenval"])
print("Scores (first 5):\n", results["scores"].head())

# 4. Evaluate and plot the first eigenfunction
# Create a grid of query points
grid = np.linspace(0, 1, 50)
X, Y = np.meshgrid(grid, grid)
query_points = np.stack([X.ravel(), Y.ravel()], axis=1) # Shape (2500, 2)

# Evaluate eigenfunctions at query points
eta_vals = results["eigenfun"](query_points)

# Plot
plt.contourf(X, Y, eta_vals[:, 0].reshape(50, 50), levels=20, cmap='RdBu')
plt.colorbar(label="Eigenfunction Value")
plt.title("First Principal Component (Eigenfunction)")
plt.show()
```

## Reuse a trained PPPCA model

When you want to persist a trained PPPCA model (eigenvalues, eigenfunctions, and
centering statistics), request the state from `pppca`, then save it.

```python
from pppca.core import pppca, save_pppca_features, load_pppca_features

results = pppca(processes, Jmax=3, return_state=True)
save_pppca_features("pppca_features.npz", state=results["state"])

# Later (or in another script)
state = load_pppca_features("pppca_features.npz")
eigenfun = state["eigenfun"]
```

## Project new samples onto existing components

For new point processes, project them onto the stored eigenfunctions to obtain
scores (kernel PCA projection on the centered Gram space).

```python
from pppca.core import project_pppca

new_scores = project_pppca(new_processes, state=state)
print(new_scores.head())
```

## Visualize eigenfunctions (low dimensions)

Use the built-in plot helper for $d \in \{1,2,3\}$ to visualize eigenfunctions.

```python
from pppca.core import plot_eigenfunctions

plot_eigenfunctions(state["eigenfun"], d=2, Jmax=3)
```

## 📄 Reference & Reproducibility

The full research paper describing the methodology is included in this repository:
- **[Read the Paper](paper/)**: See the `paper/` directory for the PDF and supplementary materials.

To reproduce the experiments and figures presented in the paper, please refer to the examples:
- **[Reproducibility Code](examples/)**: The `examples/` directory contains scripts and notebooks to generate the results and visualizations discussed in the research.