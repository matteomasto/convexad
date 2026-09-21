"""Minimal end-to-end example: BCDI phase retrieval with a population of
independent L-BFGS restarts, keeping the lowest-loss reconstruction.
"""
import time

import jax
import numpy as np

from convexad import reconstruct

# ------------------------------------------------------------------
# Load your diffraction data (replace with a real dataset).
# Iobs shape ranges from ~(100,100,100) up to ~(250,450,450) in practice.
# ------------------------------------------------------------------
Iobs = np.load("data.npz")["I"].astype(np.float32)

key = jax.random.PRNGKey(0)

t0 = time.time()
result = reconstruct(
    key,
    Iobs,
    n_restarts=16,          # population of independent random restarts
    N=64,                   # number of half-spaces
    size_factor=4.0,
    eps=0.6,                 # fixed half-space softness (no annealing)
    alpha=0.8,                # support-size penalty weight
    beta=0.1,                 # phase-TV penalty weight
    metric="mae",              # or "poisson"
    phase_type="grid",        # "grid" | "phasor" | "displacement"
    max_steps=300,
    tol=1e-6,
    memory_size=10,            # L-BFGS history depth; lower to fit more restarts
)
print(f"done in {time.time() - t0:.1f}s")

print("loss per restart:", result.all_losses)
print("steps per restart:", result.all_steps)
print("best loss:", float(result.best_loss))

support, amplitude, phase = result.evaluate(Iobs)
print("support shape:", support.shape, "amplitude:", float(amplitude))

# `result.best_params` is a plain dict pytree -- save with e.g. np.savez
# after converting leaves to numpy:
import jax.numpy as jnp  # noqa: E402
np.savez(
    "reconstruction.npz",
    n_raw=np.asarray(result.best_params["support"]["n_raw"]),
    d=np.asarray(result.best_params["support"]["d"]),
    support=np.asarray(support),
    phase=np.asarray(phase) if not isinstance(phase, tuple) else None,
)
