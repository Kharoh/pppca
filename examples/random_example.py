from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.pppca.core import pppca

# Adjust dimension here (d≥1)
d = 4

def _sample_point_processes(
    num_processes: int = 100,
    base_events: int = 20,
    jitter: float = 0.05,
    seed: int = 1234,
) -> List[torch.Tensor]:
    """Generate synthetic *d*-dim point processes for demonstration.
    Output: list of (num_points, d) tensors; each entry is a point process."""
    torch.manual_seed(seed)
    base_grid = torch.linspace(0.1, 0.9, base_events, dtype=torch.float64)
    processes: List[torch.Tensor] = []
    for _ in range(num_processes):
        # Sample number of events (stochastically drop some events)
        mask = torch.rand(base_events) > 0.25
        if not mask.any():
            mask[torch.randint(0, base_events, (1,), dtype=torch.long)] = True
        selected = base_grid[mask]
        # Make each event a random [0,1]^d point with base grid in first coordinate
        coords = [selected]
        for dd in range(1, d):
            coords.append(torch.rand_like(selected))
        events = torch.stack(coords, dim=1)  # (num_events, d)
        noise = (torch.rand_like(events) - 0.5) * jitter
        events = torch.clamp(events + noise, min=0.0, max=1.0)
        processes.append(events)
    return processes

def plot_sample_processes(processes, d, num_plot=6):
    import matplotlib.pyplot as plt
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # Import in function for 3D plot
    plt.figure(figsize=(6, 4))
    if d == 1:
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            plt.plot(X[:, 0], i + np.zeros_like(X[:, 0]), 'o-', label=f"Process {i+1}")
        plt.xlabel("x")
        plt.ylabel("Process Index")
        plt.title("Sample 1D Point Processes")
        plt.yticks(range(num_plot))
        plt.legend(loc="best")
    elif d == 2:
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            plt.plot(X[:, 0], X[:, 1], 'o-', label=f"Process {i+1}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Sample 2D Point Processes")
        plt.legend(loc="best")
    elif d == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(min(num_plot, len(processes))):
            X = processes[i].cpu().numpy()
            ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o-', label=f"Process {i+1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Sample 3D Point Processes")
        ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    """Run PPPCA_dual_multivariate on synthetic data and display diagnostic plots (first components)."""

    Jmax = 3
    sample_processes = _sample_point_processes(num_processes=25, base_events=6, seed=42)

    # Plot some sample point processes
    plot_sample_processes(sample_processes, d, num_plot=6)

    results = pppca(sample_processes, Jmax=Jmax)
    print("Eigenvalues:", results["eigenval"])
    print("Scores head:\n", results["scores"].head())
    print("Gram eigenvector coeff (first few):\n", results["coeff"][:5, :])

    # Plot leading eigenfunctions for d = 1, 2, or 3
    if d == 1:
        grid_lin = np.linspace(0, 1, 200).reshape(-1, 1)  # (200, 1)
        eta_vals = results["eigenfun"](grid_lin)  # shape (200, Jmax)
        plt.figure(figsize=(10, 4))
        for j in range(min(Jmax, 3)):
            plt.plot(grid_lin[:, 0], eta_vals[:, j], label=f"Eigenfunction {j+1}")
        plt.xlabel("x")
        plt.ylabel("Eigenfun value")
        plt.title("Leading Empirical Eigenfunctions (1D)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif d == 2:
        grid_lin = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(grid_lin, grid_lin)
        X_flat = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (10000, 2)
        eta_vals = results["eigenfun"](X_flat)  # (10000, Jmax)
        plt.figure(figsize=(12, 4))
        for j in range(min(Jmax, 3)):
            plt.subplot(1, min(Jmax, 3), j + 1)
            plt.contourf(X, Y, eta_vals[:, j].reshape(100, 100), levels=20, cmap='RdBu')
            plt.colorbar()
            plt.title(f"Eigenfunction {j+1}")
            plt.xlabel("x1")
            plt.ylabel("x2")
        plt.suptitle("Leading Empirical Eigenfunctions (2D contour)")
        plt.tight_layout()
        plt.show()
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        grid_lin = np.linspace(0, 1, 30)
        X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin)
        XYZ_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (27000, 3)
        eta_vals = results["eigenfun"](XYZ_flat)  # (27000, Jmax)
        # For 3D, show scatter for leading component
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        val = eta_vals[:, 0]
        sel = np.abs(val) > np.percentile(np.abs(val), 95)
        p = ax.scatter(XYZ_flat[sel, 0], XYZ_flat[sel, 1], XYZ_flat[sel, 2], c=val[sel], cmap='RdBu', s=8)
        fig.colorbar(p)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Leading Eigenfunction (top values, 3D)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()