import numpy as np
import pytest
import torch

import matplotlib

from pppca.core import (
    load_pppca_features,
    plot_eigenfunctions,
    pppca,
    project_pppca,
    save_pppca_features,
)


matplotlib.use("Agg")


def _sample_processes(n: int = 6, d: int = 2) -> list[torch.Tensor]:
    torch.manual_seed(0)
    processes: list[torch.Tensor] = []
    for i in range(n):
        k = 3 + (i % 3)
        processes.append(torch.rand((k, d), dtype=torch.float64))
    return processes


def test_pppca_outputs():
    processes = _sample_processes()
    results = pppca(processes, Jmax=2, return_state=True)

    assert len(results["eigenval"]) == 2
    assert results["scores"].shape == (len(processes), 2)
    assert results["coeff"].shape == (len(processes), 2)

    grid = np.random.rand(5, 2)
    eta = results["eigenfun"](grid)
    assert eta.shape == (5, 2)
    assert "row_mean" in results["state"]


def test_save_load_roundtrip(tmp_path):
    processes = _sample_processes()
    results = pppca(processes, Jmax=2, return_state=True)

    path = save_pppca_features(tmp_path / "pppca_features.npz", state=results["state"])
    state = load_pppca_features(path)

    np.testing.assert_allclose(state["eigenval"], results["state"]["eigenval"])
    assert state["coeff"].shape == results["state"]["coeff"].shape

    eta = state["eigenfun"](np.random.rand(4, 2))
    assert eta.shape == (4, 2)


def test_project_matches_training_scores():
    processes = _sample_processes()
    results = pppca(processes, Jmax=2, return_state=True)

    projected = project_pppca(processes, state=results["state"], work_dtype=torch.float64)
    np.testing.assert_allclose(
        projected.values,
        results["scores"].values,
        rtol=1e-4,
        atol=1e-6,
    )


def test_plot_eigenfunctions_smoke():
    processes = _sample_processes()
    results = pppca(processes, Jmax=1, return_state=True)

    plot_eigenfunctions(results["eigenfun"], d=1, Jmax=1, grid_size=20)
    plot_eigenfunctions(results["eigenfun"], d=2, Jmax=1, grid_size=10)

    with pytest.raises(ValueError):
        plot_eigenfunctions(results["eigenfun"], d=4)
