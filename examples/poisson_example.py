"""Example: PP-PCA on 2D homogeneous Poisson point processes.

This script draws independent realisations of a homogeneous Poisson point process
on the unit square, runs the `pppca` routine, and compares the empirical
spectral quantities with the closed-form formulas recalled in
`parts/separable_processes.md`.
"""

from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from src import pppca


def sample_homogeneous_poisson_processes_2d(
    num_processes: int,
    intensity: float,
    seed: int | None = None,
) -> List[torch.Tensor]:
    """Generate i.i.d. PPP realisations on [0,1]^2 with constant intensity λ."""
    rng = np.random.default_rng(seed)
    out: List[torch.Tensor] = []
    for _ in range(num_processes):
        count = rng.poisson(intensity)
        if count == 0:
            out.append(torch.empty((0, 2), dtype=torch.float64))
        else:
            pts = rng.uniform(0.0, 1.0, size=(count, 2)).astype(np.float64, copy=False)
            out.append(torch.from_numpy(pts))
    return out


def eigenvalue_1d_unit(j: int) -> float:
    # 1D eigenvalue for kernel min(s,t) with λ=1
    return 4.0 / (((2*j - 1) * np.pi) ** 2)

def eigenvalue_2d(intensity: float, j1: int, j2: int) -> float:
    # Correct: single global λ in 2D
    return intensity * eigenvalue_1d_unit(j1) * eigenvalue_1d_unit(j2)

# Optional: general dD
def eigenvalue_d(intensity: float, js: Sequence[int]) -> float:
    val = 1.0
    for j in js:
        val *= eigenvalue_1d_unit(j)
    return intensity * val


def eigenfunction_1d(j: int, t: np.ndarray) -> np.ndarray:
    """Normalised 1D eigenfunction φ_j(t) = √2 sin(((2j-1)π t)/2)."""
    return np.sqrt(2.0) * np.sin(((2 * j - 1) * np.pi * t) / 2.0)


def eigenfunction_2d(j1: int, j2: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tensor product φ_{j1,j2}(x,y) = φ_{j1}(x) φ_{j2}(y)."""
    return eigenfunction_1d(j1, x) * eigenfunction_1d(j2, y)


def top_theoretical_modes(
    intensity: float,
    num_modes: int,
    per_dim: int | None = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Return the strongest `num_modes` theoretical pairs (j1,j2) and λ-values."""
    if per_dim is None:
        per_dim = max(3, int(np.ceil(np.sqrt(num_modes))) + 2)
    candidates = [
        (j1, j2) for j1 in range(1, per_dim + 1) for j2 in range(1, per_dim + 1)
    ]
    candidates.sort(key=lambda pair: eigenvalue_2d(intensity, *pair), reverse=True)
    selected = candidates[:num_modes]
    eigenvals = np.array([eigenvalue_2d(intensity, *pair) for pair in selected])
    return selected, eigenvals


def evaluate_empirical_eigenfunctions(
    eigenfun_callable,
    grid_lin: np.ndarray,
    num_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate empirical eigenfunctions returned by PP-PCA on a tensor grid."""
    xx, yy = np.meshgrid(grid_lin, grid_lin, indexing="xy")
    grid_flat = np.column_stack([xx.ravel(), yy.ravel()])
    values = eigenfun_callable(grid_flat)
    if values.shape[1] < num_components:
        raise ValueError("Not enough empirical eigenfunctions to evaluate")
    vals = values[:, :num_components].reshape(xx.shape + (num_components,))
    return xx, yy, vals


def evaluate_theoretical_eigenfunctions(
    intensity: float,
    combos: Sequence[Tuple[int, int]],
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Evaluate theoretical eigenfunctions on the same tensor grid."""
    stack = [eigenfunction_2d(j1, j2, xx, yy) for (j1, j2) in combos]
    return np.stack(stack, axis=-1)


def plot_sample_processes(processes: Sequence[torch.Tensor], num_examples: int = 6) -> plt.Figure | None:
    """Display a few sampled point patterns."""
    cols = min(num_examples, len(processes))
    if cols == 0:
        return None
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2), sharex=True, sharey=True)
    if cols == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        pts = processes[idx].cpu().numpy()
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color="tab:blue")
        else:
            ax.text(0.5, 0.5, "no points", ha="center", va="center")
        ax.set_title(f"Process {idx + 1}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
    fig.suptitle("Homogeneous PPP samples on [0,1]^2")
    fig.tight_layout()
    return fig


def plot_eigenvalues(empirical: np.ndarray, theoretical: np.ndarray) -> plt.Figure:
    """Compare empirical and theoretical eigenvalues on a log-scale plot."""
    idx = np.arange(1, empirical.size + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(idx, empirical, "o-", label="Empirical")
    ax.plot(idx, theoretical, "s--", label="Theoretical")
    ax.set_yscale("log")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("PPPCA eigenvalues vs theory")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_empirical_eigenfunctions(
    xx: np.ndarray,
    yy: np.ndarray,
    empirical_vals: np.ndarray,
) -> List[plt.Figure]:
    figures: List[plt.Figure] = []
    num_components = empirical_vals.shape[2]
    for comp in range(num_components):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cf = ax.contourf(xx, yy, empirical_vals[..., comp], levels=51, cmap="RdBu")
        ax.set_title(f"Empirical eigenfunction {comp + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(cf)
        fig.tight_layout()
        figures.append(fig)
    return figures


def compute_adaptive_threshold(
    empirical_eigenvalues: np.ndarray,
    theoretical_eigenvalues: np.ndarray,
) -> float:
    """Estimate a clustering threshold from available eigenvalues."""
    values = np.concatenate(
        [
            np.asarray(empirical_eigenvalues, dtype=float).ravel(),
            np.asarray(theoretical_eigenvalues, dtype=float).ravel(),
        ]
    )
    values = values[np.isfinite(values)]
    if values.size < 2:
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        return max(scale * 1e-6, 1e-12)
    sorted_vals = np.sort(values)[::-1]
    diffs = np.abs(np.diff(sorted_vals))
    positive_diffs = diffs[diffs > 0.0]
    scale = float(np.max(np.abs(values)))
    if positive_diffs.size == 0:
        return max(scale * 1e-6, 1e-12)
    baseline = float(np.percentile(positive_diffs, 25.0)) if positive_diffs.size > 1 else float(positive_diffs[0])
    threshold = baseline * 0.5
    return max(threshold, scale * 1e-6)


def group_indices_by_threshold(
    eigenvalues: Sequence[float],
    threshold: float,
) -> List[List[int]]:
    indices = list(range(len(eigenvalues)))
    if not indices:
        return []
    groups: List[List[int]] = [[indices[0]]]
    for prev_idx, current_idx in zip(indices, indices[1:]):
        if abs(eigenvalues[current_idx] - eigenvalues[prev_idx]) <= threshold:
            groups[-1].append(current_idx)
        else:
            groups.append([current_idx])
    return groups


def plot_eigenfunction_groups(
    xx: np.ndarray,
    yy: np.ndarray,
    grid_lin: np.ndarray,
    combos: Sequence[Tuple[int, int]],
    theoretical_vals: np.ndarray,
    empirical_vals: np.ndarray,
    groups: Sequence[Sequence[int]],
    theoretical_eigenvalues: np.ndarray,
) -> List[Tuple[plt.Figure, Tuple[int, ...]]]:
    if grid_lin.size > 1:
        dx = float(grid_lin[1] - grid_lin[0])
    else:
        dx = 1.0
    dy = dx
    dxdy = dx * dy
    figures: List[Tuple[plt.Figure, Tuple[int, ...]]] = []
    for group in groups:
        include_span = len(group) > 1
        num_cols = len(group) + (1 if include_span else 0)
        fig, axes = plt.subplots(
            2,
            num_cols,
            figsize=(4.8 * num_cols, 7.0),
            sharex=True,
            sharey=True,
        )
        if isinstance(axes, np.ndarray) and axes.ndim == 1:
            axes = axes[:, np.newaxis]
        for col_idx, comp_idx in enumerate(group):
            theo_block = theoretical_vals[..., comp_idx]
            emp_block = empirical_vals[..., comp_idx]
            vmax = max(float(np.max(np.abs(theo_block))), float(np.max(np.abs(emp_block))))
            levels = np.linspace(-vmax, vmax, 51) if vmax > 0.0 else 51
            cf_theo = axes[0, col_idx].contourf(xx, yy, theo_block, levels=levels, cmap="RdBu")
            cf_emp = axes[1, col_idx].contourf(xx, yy, emp_block, levels=levels, cmap="RdBu")
            axes[0, col_idx].set_title(f"Theory ({combos[comp_idx][0]},{combos[comp_idx][1]})")
            axes[1, col_idx].set_title(f"Empirical #{comp_idx + 1}")
            axes[1, col_idx].set_xlabel("x")
            if col_idx == 0:
                axes[0, col_idx].set_ylabel("y")
                axes[1, col_idx].set_ylabel("y")
            fig.colorbar(cf_theo, ax=axes[:, col_idx], fraction=0.046, pad=0.04)
        if include_span:
            span_idx = num_cols - 1
            theo_span = np.sum(theoretical_vals[..., group], axis=-1)
            emp_span = np.sum(empirical_vals[..., group], axis=-1)
            theo_norm = float(np.sum(theo_span**2) * dxdy)
            emp_norm = float(np.sum(emp_span**2) * dxdy)
            if theo_norm > 0.0:
                theo_span = theo_span / np.sqrt(theo_norm)
            if emp_norm > 0.0:
                emp_span = emp_span / np.sqrt(emp_norm)
            vmax = max(float(np.max(np.abs(theo_span))), float(np.max(np.abs(emp_span))))
            levels = np.linspace(-vmax, vmax, 51) if vmax > 0.0 else 51
            cf_theo = axes[0, span_idx].contourf(xx, yy, theo_span, levels=levels, cmap="RdBu")
            cf_emp = axes[1, span_idx].contourf(xx, yy, emp_span, levels=levels, cmap="RdBu")
            axes[0, span_idx].set_title("Theory span")
            axes[1, span_idx].set_title("Empirical span")
            axes[1, span_idx].set_xlabel("x")
            if span_idx == 0:
                axes[0, span_idx].set_ylabel("y")
                axes[1, span_idx].set_ylabel("y")
            fig.colorbar(cf_theo, ax=axes[:, span_idx], fraction=0.046, pad=0.04)
        eigenvals_text = ", ".join(f"{theoretical_eigenvalues[idx]:.3e}" for idx in group)
        idx_text = ", ".join(str(idx + 1) for idx in group)
        fig.suptitle(
            f"Eigenfunctions comparison – indices [{idx_text}], λ_theo≈[{eigenvals_text}]"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figures.append((fig, tuple(group)))
    return figures


def save_figures(figures: Iterable[Tuple[plt.Figure, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, filename in figures:
        target = output_dir / filename
        fig.savefig(target, dpi=300)
        print(f"Saved figure to {target}")


def main() -> None:
    """Entry point for the homogeneous PPP comparison demo."""
    intensity = 100.0          # λ: expected number of points per process on [0,1]^2
    num_processes = 500        # number of independent realisations
    jmax = 6                   # number of components to keep in PP-PCA
    grid_size = 75             # resolution for function visualisation
    seed = 42

    print(
        "Running PPPCA on 2D homogeneous Poisson processes with "
        f"λ={intensity}, n={num_processes}, Jmax={jmax}."
    )

    processes = sample_homogeneous_poisson_processes_2d(
        num_processes=num_processes,
        intensity=intensity,
        seed=seed,
    )

    figures_to_save: List[Tuple[plt.Figure, str]] = []

    sample_fig = plot_sample_processes(processes, num_examples=6)
    if sample_fig is not None:
        figures_to_save.append((sample_fig, "poisson_samples.png"))

    results = pppca(processes, Jmax=jmax)
    empirical_eigenvalues = np.asarray(results["eigenval"], dtype=float)

    combos, theoretical_eigenvalues = top_theoretical_modes(intensity, jmax)

    grid_lin = np.linspace(0.0, 1.0, grid_size)
    xx, yy, empirical_vals = evaluate_empirical_eigenfunctions(
        results["eigenfun"], grid_lin, jmax
    )
    theoretical_vals = evaluate_theoretical_eigenfunctions(
        intensity, combos, xx, yy
    )

    print("\nTheoretical modes (sorted by eigenvalue):")
    for idx, ((j1, j2), val) in enumerate(zip(combos, theoretical_eigenvalues), start=1):
        print(f"  {idx:2d}: (j1,j2)=({j1},{j2}), λ_theo={val:.6e}")

    threshold = compute_adaptive_threshold(empirical_eigenvalues, theoretical_eigenvalues)
    groups = group_indices_by_threshold(theoretical_eigenvalues, threshold)

    print(f"\nAdaptive eigenvalue grouping threshold: {threshold:.6e}")
    for group in groups:
        idx_text = ", ".join(str(idx + 1) for idx in group)
        eigenvals_text = ", ".join(f"{theoretical_eigenvalues[idx]:.6e}" for idx in group)
        combos_text = ", ".join(f"({combos[idx][0]},{combos[idx][1]})" for idx in group)
        print(f"  Group [{idx_text}] modes {combos_text} ⇒ λ_theo≈[{eigenvals_text}]")

    eigenvalue_fig = plot_eigenvalues(empirical_eigenvalues, theoretical_eigenvalues)
    figures_to_save.append((eigenvalue_fig, "poisson_eigenvalues.png"))

    empirical_mode_figures = plot_empirical_eigenfunctions(
        xx,
        yy,
        empirical_vals,
    )
    for comp_idx, fig in enumerate(empirical_mode_figures, start=1):
        figures_to_save.append((fig, f"poisson_empirical_mode_{comp_idx}.png"))

    group_figures = plot_eigenfunction_groups(
        xx,
        yy,
        grid_lin,
        combos,
        theoretical_vals,
        empirical_vals,
        groups,
        theoretical_eigenvalues,
    )

    for group_idx, (fig, group) in enumerate(group_figures, start=1):
        component_label = "-".join(str(idx + 1) for idx in group)
        filename = f"poisson_eigen_group_{group_idx}_components_{component_label}.png"
        figures_to_save.append((fig, filename))

    images_dir = Path(__file__).resolve().parent.parent / "images"
    save_figures(figures_to_save, images_dir)

    plt.show()


if __name__ == "__main__":
    main()
