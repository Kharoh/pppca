Usage
=====

Quickstart
----------

.. code-block:: python

   import torch
   from pppca.core import pppca

   processes = [torch.rand((5, 2)) for _ in range(10)]
   results = pppca(processes, Jmax=2)
   print(results["eigenval"])

Save and reload models
----------------------

.. code-block:: python

   from pppca.core import load_pppca_features, pppca, save_pppca_features

   results = pppca(processes, Jmax=3, return_state=True)
   save_pppca_features("pppca_features.npz", state=results["state"])

   state = load_pppca_features("pppca_features.npz")
   eigenfun = state["eigenfun"]

Project new samples
-------------------

.. code-block:: python

   from pppca.core import project_pppca

   new_scores = project_pppca(processes, state=state)
   print(new_scores.head())

Visualize eigenfunctions
------------------------

.. code-block:: python

   from pppca.core import plot_eigenfunctions

   plot_eigenfunctions(state["eigenfun"], d=2, Jmax=3)
