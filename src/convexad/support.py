from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def make_coords(grid_shape):
    D, H, W = grid_shape
    z = jnp.linspace(-(D - 1) / 2.0, (D - 1) / 2.0, D)
    y = jnp.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, H)
    x = jnp.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, W)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    return jnp.stack([xx, yy, zz], axis=-1)  # (D, H, W, 3)


def _sigma_i(n_i, d_i, coords, eps):
    dot = jnp.einsum("dhwc,c->dhw", coords, n_i)
    return jax.nn.sigmoid((d_i - dot) / eps)


@jax.custom_vjp
def halfspace_support(n, d, coords, eps):
    """Soft polytope indicator, memory O(D*H*W) instead of O(D*H*W*N)."""
    logS0 = jnp.zeros(coords.shape[:3], dtype=coords.dtype)

    def step(logS, nd_i):
        n_i, d_i = nd_i
        sigma = _sigma_i(n_i, d_i, coords, eps)
        logS = logS + jnp.log(jnp.clip(sigma, 1e-6, 1.0))
        return logS, None

    logS, _ = lax.scan(step, logS0, (n, d))
    return jnp.exp(logS)


def _halfspace_support_fwd(n, d, coords, eps):
    S = halfspace_support(n, d, coords, eps)
    return S, (n, d, coords, eps, S)


def _halfspace_support_bwd(res, g):
    n, d, coords, eps, S = res
    gS = g * S

    def step(dcoords_acc, nd_i):
        n_i, d_i = nd_i
        sigma = _sigma_i(n_i, d_i, coords, eps)
        active = jnp.logical_and(sigma > 1e-6, sigma < 1.0).astype(sigma.dtype)
        w = gS * active * (1.0 - sigma) / eps
        dd_i = jnp.sum(w)
        dn_i = -jnp.einsum("dhw,dhwc->c", w, coords)
        dcoords_acc = dcoords_acc - w[..., None] * n_i
        return dcoords_acc, (dn_i, dd_i)

    dcoords0 = jnp.zeros_like(coords)
    dcoords, (dn, dd) = lax.scan(step, dcoords0, (n, d))
    return dn, dd, dcoords, None


halfspace_support.defvjp(_halfspace_support_fwd, _halfspace_support_bwd)


def stereographic_to_unit(p):
    """Inverse stereographic projection: R^2 -> S^2 minus the north pole.

    p : (..., 2) -> n : (..., 3), unit norm.
    """
    p2 = jnp.sum(p ** 2, axis=-1, keepdims=True)
    x = 2.0 * p[..., 0:1]
    y = 2.0 * p[..., 1:2]
    z = p2 - 1.0
    return jnp.concatenate([x, y, z], axis=-1) / (1.0 + p2)


def unit_to_stereographic(n, eps=1e-6):
    """Forward stereographic projection: S^2 minus the north pole -> R^2.

    Only used for initialization (mapping a randomly sampled unit vector
    to its (p1, p2) coordinates); not needed in the training loop.
    """
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    denom = jnp.clip(1.0 - z, eps, None)
    return jnp.stack([x / denom, y / denom], axis=-1)


def init_support_params(key, N, grid_shape, size_factor=4.0):
    """Random init for a single instance's half-space support parameters."""
    key_n, key_d = jax.random.split(key)
    n0 = jax.random.normal(key_n, (N, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    p0 = unit_to_stereographic(n0)
    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    d0 = jnp.ones((N,)) * R
    return {"p_raw": p0, "d": d0}


def compute_support(params, coords, eps):
    """Map the minimal 2-DOF parameterization to a unit normal and evaluate S."""
    n = stereographic_to_unit(params["p_raw"])
    return halfspace_support(n, params["d"], coords, eps)
