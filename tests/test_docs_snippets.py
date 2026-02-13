import numpy as np
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


def _snippet_processes(n: int = 10, d: int = 2) -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.rand((5, d), dtype=torch.float64) for _ in range(n)]


def test_docs_quickstart_snippet():
    processes = _snippet_processes()
    results = pppca(processes, Jmax=2)

    assert len(results["eigenval"]) == 2
    assert results["scores"].shape == (len(processes), 2)


def test_docs_save_reload_snippet(tmp_path):
    processes = _snippet_processes()
    results = pppca(processes, Jmax=3, return_state=True)

    path = save_pppca_features(tmp_path / "pppca_features.npz", state=results["state"])
    state = load_pppca_features(path)
    eigenfun = state["eigenfun"]

    eta = eigenfun(np.random.rand(6, 2))
    assert eta.shape == (6, 3)


def test_docs_project_new_samples_snippet():
    processes = _snippet_processes()
    results = pppca(processes, Jmax=3, return_state=True)

    new_scores = project_pppca(processes, state=results["state"], work_dtype=torch.float64)
    assert new_scores.shape == (len(processes), 3)


def test_docs_visualize_eigenfunctions_snippet(tmp_path):
    processes = _snippet_processes()
    results = pppca(processes, Jmax=3, return_state=True)

    path = save_pppca_features(tmp_path / "pppca_features.npz", state=results["state"])
    state = load_pppca_features(path)

    plot_eigenfunctions(state["eigenfun"], d=2, Jmax=3, grid_size=12)