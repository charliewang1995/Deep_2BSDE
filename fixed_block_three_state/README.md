# Shared-time geometric 2BSDE experiment

This package extends the earlier reusable `two_bsde` implementation without removing the classical architecture.

New components:

- `GeometricTimeConditionedBlock` in `two_bsde/networks.py`: one shared neural map `(t, state) -> (A, Gamma)`.
- `GeometricSharedTimeFixedAux2BSDE` in `two_bsde/model.py`: uses that same block at every time step, including `t=0`.
- `shared_time_three_state_test.ipynb`: the same controlled three-state analytic benchmark, with direct diagnostics for `A` and `Gamma` along fresh paths.

The notebook deliberately uses `u_aux = 1` so that the posterior diffusion is excited.  In this benchmark `u_aux = 0` freezes the posterior and makes the posterior Hessian very weakly identifiable from the 2BSDE recursion.
