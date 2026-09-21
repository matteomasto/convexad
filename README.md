# ConvexAD

<img width="1240" height="820" alt="image" src="https://github.com/user-attachments/assets/df4366b5-9a40-466b-a82e-a13eff3cfae0" />

---
Geometrically regularized automatic differentiation framework for BCDI phase retrieval.
---

1. **A hand-derived, memory-efficient adjoint for the half-space support op**,
   replacing the naive `(D, H, W, N)`-materializing version.
2. **The unit-sphere constraint on half-space normals is removed**, not
   ported. **Revision:** an earlier version of this parameterization used
   `n = n_raw / ||n_raw||` with a free `n_raw in R^3` (3 parameters for a
   2-DOF constraint). This is mathematically valid but leaves a "gauge"
   direction with exactly zero gradient and zero curvature; profiling an
   actual `optax.lbfgs` run showed `||n_raw||` drifting ~3.7x over 80 steps
   from numerical mixing alone, which silently shrinks the *useful*
   gradient (it scales as `1/||n_raw||`) and degrades L-BFGS's curvature
   estimate. This is fixed by using a minimal, non-redundant 2-DOF
   parameterization (inverse stereographic projection) with no flat
   direction to drift into -- see the derivation comment at the top of
   `support.py`.

## Why a custom VJP for the half-space support

`S(x) = prod_i sigmoid((d_i - n_i . x) / eps)`. The direct implementation
computes `dot = einsum('dhwc,nc->dhwn', coords, n)`, a `(D, H, W, N)` tensor,
and autodiff would keep something of that shape alive for the backward pass.
At this project's largest problem size (`Iobs` up to `250x450x450`, so a grid
of `125x225x225 ~= 6.33M` voxels, `N` up to `256`):

```
6.33M voxels * 256 planes * 4 bytes ~= 6.0 GiB   -- for ONE tensor
```

which does not fit a 32 GB GPU budget once FFT buffers and L-BFGS history are
also resident. `src/convexad/support.py` instead `lax.scan`s over the
`N` half-spaces, keeping only a running `(D, H, W)` log-support accumulator,
and recomputes `sigma_i` per-plane in the backward pass rather than storing
it. Peak extra memory becomes `O(D*H*W)`, independent of `N`. The analytic
gradient is verified against finite differences in the test in this repo's
history (see the derivation comment at the top of `support.py`).

The Fourier data-fidelity term is *not* hand-differentiated: `jnp.fft` is
linear, so JAX's built-in VJP is already an (adjoint) FFT with no extra
activation storage -- there is nothing to improve there.


## Package layout

```
src/convexad/
    support.py    # HalfSpaceSupport equivalent + custom_vjp
    phase.py      # GridPhase / GridPhasor / DisplacementPhasor equivalents
    losses.py     # mae, poisson_kl, fourier_loss, tv_loss_phase, total_loss
    model.py      # single-instance init / forward / loss_fn
    optimize.py   # population init + vmapped optax.lbfgs driver
examples/
    run_reconstruction.py
```

Everything in `support.py`, `phase.py`, `losses.py`, and `model.py` operates
on a **single reconstruction instance** (no leading batch axis) -- the
population axis is added purely by `jax.vmap` in `optimize.py`. This keeps
the physics/model code simple and lets `vmap` (rather than hand-written
batch axes everywhere) handle the population dimension, including through
the custom VJP, which composes with `vmap` without modification.

## Installation

```bash
pip install -e .
```

CPU-only is enough to run the test/example at small sizes; for real BCDI
grid sizes you'll want a CUDA-enabled `jax[cuda12]` install.

## Usage

```python
import jax
import numpy as np
from convexad import reconstruct

Iobs = np.load("data.npz")["I"].astype(np.float32)

result = reconstruct(
    jax.random.PRNGKey(0), Iobs,
    n_restarts=16, N=64, eps=0.6, alpha=0.8, beta=0.1,
    metric="mae", phase_type="grid",
    max_steps=300, tol=1e-6, memory_size=10,
)
support, amplitude, phase = result.evaluate(Iobs)
```

See `examples/run_reconstruction.py` for a complete script.


