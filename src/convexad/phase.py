# =============================================================================
# PHASE PARAMETERIZATION
# =============================================================================
# Single-instance (no leading batch axis); see support.py for why.
import jax
import jax.numpy as jnp

_TWO_PI = 2.0 * jnp.pi


def _reciprocal_vector(hkl, lattice_matrix):
    hkl = jnp.asarray(hkl, dtype=jnp.float32)
    if lattice_matrix is not None:
        A = jnp.asarray(lattice_matrix, dtype=jnp.float32)
        B = _TWO_PI * jnp.transpose(jnp.linalg.inv(A))
        Q = B @ hkl
    else:
        Q = _TWO_PI * hkl
    return jnp.linalg.norm(Q)


def init_phase_params(key, grid_shape, phase_type="grid", initial_guess=None,
                       hkl=(1, 1, 1), lattice_matrix=None, init_scale=1e-3):
    """Initialize phase parameters and any static (non-trainable) metadata.

    Returns
    -------
    params : dict of trainable arrays for this phase type.
    static : dict of non-trainable metadata (e.g. Qnorm for 'displacement').
    """
    grid_shape = tuple(grid_shape)

    if phase_type == "grid":
        if initial_guess is not None:
            phase0 = jnp.angle(initial_guess)
        else:
            phase0 = 1e-3*jax.random.uniform(key, grid_shape, minval=0.0, maxval=1.0)
        return {"phase": phase0}, {"phase_type": "grid"}

    if phase_type == "phasor":
        return (
            {"c_raw": jnp.ones(grid_shape), "s_raw": jnp.zeros(grid_shape)},
            {"phase_type": "phasor", "eps": 1e-8},
        )

    if phase_type == "displacement":
        Qnorm = _reciprocal_vector(hkl, lattice_matrix)
        if initial_guess is not None:
            phase0 = jnp.angle(initial_guess)
            u0 = phase0 / (Qnorm + 1e-8)
        else:
            u0 = init_scale * jax.random.normal(key, grid_shape)
        return {"u": u0}, {"phase_type": "displacement", "Qnorm": Qnorm}

    raise ValueError(f"Unknown phase_type: {phase_type!r}")


def compute_phase(params, static):
    """Returns a scalar phase field phi, shape (D, H, W)."""
    phase_type = static["phase_type"]
    if phase_type == "grid":
        return params["phase"]
    if phase_type == "displacement":
        return static["Qnorm"] * params["u"]
    raise ValueError(f"{phase_type!r} has no scalar phase; use compute_phasor.")


def compute_phasor(params, static):
    """Returns a (cos, sin) pair, each shape (D, H, W)."""
    phase_type = static["phase_type"]
    if phase_type == "phasor":
        c, s = params["c_raw"], params["s_raw"]
        denom = jnp.sqrt(c ** 2 + s ** 2 + static["eps"])
        return c / denom, s / denom
    if phase_type in ("grid", "displacement"):
        phi = compute_phase(params, static)
        return jnp.cos(phi), jnp.sin(phi)
    raise ValueError(f"Unknown phase_type: {phase_type!r}")
