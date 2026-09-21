# =============================================================================
# MULTI-PART (NON-CONVEX) SUPPORT
# =============================================================================
# Non-convex support as a soft UNION of M convex parts, each a soft
# INTERSECTION of N_per_part half-spaces -- ported from a draft TensorFlow
# `MultiHalfSpaceMaskSupport`, with the same fixes already applied to the
# single-convex-part support in `support.py`:
#
#   - Minimal (2-DOF) stereographic normal parameterization instead of
#     spherical angles (theta, phi). Spherical angles have a coordinate
#     singularity at the poles (dn/dtheta -> 0 as theta -> 0 or pi) and a
#     latitude-dependent "speed" (the Jacobian scales with sin(theta)),
#     which reintroduces exactly the optimizer-conditioning problem fixed
#     for the single-part case in support.py -- just via a different
#     mechanism (coordinate-chart distortion instead of a redundant flat
#     direction). `theta, phi` in the original are replaced by `p_raw`
#     (M, N, 2) here, mapped through the same `stereographic_to_unit` used
#     for the single-part support.
#   - The per-part intersection reuses `halfspace_support` ITSELF (not a
#     new bespoke implementation) via `jax.vmap` over the M parts. This
#     means every part automatically gets: the scan-based O(D,H,W) memory
#     footprint (independent of N_per_part), and the corrected
#     clip-boundary gradient mask (see the comment in
#     `_halfspace_support_bwd`). Re-deriving a separate (D,H,W,M,N)
#     custom VJP for this file would have risked reintroducing both bugs.
#
# The union of M parts, `1 - prod_m(1 - S_m)`, is left to ordinary
# autodiff: M is small (a handful of domains) -- unlike N, which can be in
# the hundreds -- so materializing a (D,H,W,M) tensor for the union step is
# not the kind of memory concern the (D,H,W,N) tensor was for planes.
import jax
import jax.numpy as jnp

from .support import halfspace_support, stereographic_to_unit, unit_to_stereographic


def init_multi_support_params(key, M_parts, N_per_part, grid_shape, size_factor=4.0,
                               use_gates=False):
    """Random init for a single instance's multi-part support parameters."""
    key_n, key_d, key_c, key_g = jax.random.split(key, 4)

    n0 = jax.random.normal(key_n, (M_parts, N_per_part, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    p0 = unit_to_stereographic(n0)

    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    d0 = jnp.ones((M_parts, N_per_part)) * R

    centers0 = jax.random.uniform(key_c, (M_parts, 3), minval=-R, maxval=R)

    params = {"p_raw": p0, "d": d0, "centers": centers0}
    if use_gates:
        # sigmoid(1.0) ~= 0.73: mostly-open gates at init, matching the
        # original's `tf.ones(...)` init for the pre-sigmoid gate logit.
        params["alpha_raw"] = jnp.ones((M_parts,))
    return params


def compute_multi_support(params, coords, eps, use_gates=False):
    """Soft non-convex support: union of M soft convex (half-space
    intersection) parts, each optionally centered independently.

    Parameters
    ----------
    params : dict with "p_raw" (M,N,2), "d" (M,N), "centers" (M,3), and
        optionally "alpha_raw" (M,) if use_gates=True.
    coords : (D, H, W, 3), shared fixed coordinate grid.
    eps : half-space softness.

    Returns
    -------
    support : (D, H, W) array in [0, 1].
    """
    n = stereographic_to_unit(params["p_raw"])          # (M, N, 3)
    d = params["d"]                                       # (M, N)
    centers = params["centers"]                            # (M, 3)

    # (M, D, H, W, 3): each part sees the grid shifted by its own center.
    coords_shifted = coords[None, ...] - centers[:, None, None, None, :]

    def per_part(n_m, d_m, coords_m):
        return halfspace_support(n_m, d_m, coords_m, eps)   # (D, H, W)

    S_parts = jax.vmap(per_part, in_axes=(0, 0, 0))(n, d, coords_shifted)  # (M,D,H,W)

    if use_gates:
        gate = jax.nn.sigmoid(params["alpha_raw"])[:, None, None, None]  # (M,1,1,1)
        S_parts = S_parts * gate

    # Soft union (probabilistic OR): 1 - prod_m(1 - S_m).
    support = 1.0 - jnp.prod(1.0 - S_parts, axis=0)
    return support
