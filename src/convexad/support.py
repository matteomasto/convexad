import math
import jax
import jax.numpy as jnp
from jax import lax

_TINY = 1e-6
_LOG_TINY = math.log(_TINY)


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
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


def _log_support(n, d, coords, eps):
    """log S = sum_i log sigma_i, O(DHW) memory."""
    def step(logS, nd_i):
        n_i, d_i = nd_i
        s = _sigma_i(n_i, d_i, coords, eps)
        return logS + jnp.log(jnp.clip(s, _TINY, 1.0)), None

    logS0 = jnp.zeros(coords.shape[:3], dtype=coords.dtype)
    logS, _ = lax.scan(step, logS0, (n, d))
    return logS


# ----------------------------------------------------------------------------
# Soft polytope indicator with a memory-light custom VJP
# ----------------------------------------------------------------------------
@jax.custom_vjp
def _halfspace_support(n, d, coords, eps, bwd_scale):
    return jnp.exp(_log_support(n, d, coords, eps))


def _fwd(n, d, coords, eps, bwd_scale):
    logS = _log_support(n, d, coords, eps)
    return jnp.exp(logS), (n, d, coords, eps, bwd_scale, logS)


def _bwd(res, g):
    n, d, coords, eps, k, logS = res
    eps_b = eps * k
    flat_coords = coords.reshape(-1, coords.shape[-1])

    def per_plane(nd_i):
        n_i, d_i = nd_i
        arg = d_i - jnp.einsum("dhwc,c->dhw", coords, n_i)
        s_f = jax.nn.sigmoid(arg / eps)          # forward sigmoid
        s_b = jax.nn.sigmoid(arg / eps_b)        # (possibly wider) backward one
        active = (s_f > _TINY).astype(s_f.dtype)  # forward clip => zero grad
        # leave-one-out support S_{-i} = S / sigma_i, done in log domain
        loo = jnp.exp(logS - jnp.log(jnp.clip(s_f, _TINY, 1.0)))
        w = g * loo * active * s_b * (1.0 - s_b) / eps_b
        dd = jnp.sum(w)
        dn = -(w.reshape(-1) @ flat_coords)
        return dn, dd

    dn, dd = lax.map(per_plane, (n, d))
    # coords / eps / bwd_scale are constants: no cotangent, no (D,H,W,3) buffer
    return dn, dd, None, None, None


_halfspace_support.defvjp(_fwd, _bwd)


def halfspace_support(n, d, coords, eps, bwd_scale=1.0):
    """Soft polytope indicator, memory O(D*H*W).

    bwd_scale > 1 widens the sigmoid in the backward pass only, so planes that
    are slightly outside the object still receive a (biased) gradient. Use
    ~2-3 early in training and anneal to 1.0 (exact gradient).
    """
    return _halfspace_support(n, d, coords, eps,
                              jnp.asarray(bwd_scale, coords.dtype))


# ----------------------------------------------------------------------------
# Normal parameterizations
# ----------------------------------------------------------------------------
def stereographic_to_unit(p):
    """R^2 -> S^2 minus the north pole. p: (..., 2) -> n: (..., 3)."""
    p2 = jnp.sum(p ** 2, axis=-1, keepdims=True)
    x = 2.0 * p[..., 0:1]
    y = 2.0 * p[..., 1:2]
    z = p2 - 1.0
    return jnp.concatenate([x, y, z], axis=-1) / (1.0 + p2)


def unit_to_stereographic(n, eps=1e-6):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    denom = jnp.clip(1.0 - z, eps, None)
    return jnp.stack([x / denom, y / denom], axis=-1)


def normals_from_params(params):
    """Unit normals. 'n_raw' (default, no singularity) or legacy 'p_raw'."""
    if "n_raw" in params:
        v = params["n_raw"]
        return v / jnp.sqrt(jnp.sum(v ** 2, axis=-1, keepdims=True) + 1e-12)
    return stereographic_to_unit(params["p_raw"])


def _with_normals(params, n_new):
    out = dict(params)
    if "n_raw" in params:
        out["n_raw"] = n_new
    else:
        out["p_raw"] = unit_to_stereographic(n_new)
    return out


def init_support_params(key, N, grid_shape, size_factor=4.0):
    key_n, _ = jax.random.split(key)
    n0 = jax.random.normal(key_n, (N, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    return {"n_raw": n0, "d": jnp.ones((N,)) * R}


def compute_support(params, coords, eps, bwd_scale=1.0):
    return halfspace_support(normals_from_params(params), params["d"],
                             coords, eps, bwd_scale)


# ----------------------------------------------------------------------------
# Diagnostics: which planes are dead / redundant / killers?
# ----------------------------------------------------------------------------
def plane_health(params, coords, eps):
    """
    unique_cut[i] in [0,1]: fraction of the volume that would exist WITHOUT
        plane i which plane i alone removes.
        ~0  -> dead (outside the object) or shadowed by other planes
        ~1  -> killer (removes almost everything the others leave)
    nearest_cos[i]: max cosine similarity with any other plane (redundancy).
    """
    n = lax.stop_gradient(normals_from_params(params))
    d = lax.stop_gradient(params["d"])
    logS = _log_support(n, d, coords, eps)
    vol = jnp.sum(jnp.exp(logS))

    def per_plane(nd_i):
        n_i, d_i = nd_i
        s = _sigma_i(n_i, d_i, coords, eps)
        loo = jnp.exp(logS - jnp.log(jnp.clip(s, _TINY, 1.0)))
        return jnp.sum(loo * (1.0 - s)), jnp.sum(loo)

    excl, vol_wo = lax.map(per_plane, (n, d))
    unique_cut = excl / jnp.maximum(vol_wo, 1e-8)

    N = n.shape[0]
    cos = jnp.where(jnp.eye(N, dtype=bool), -1.0, n @ n.T)
    return {"vol": vol, "unique_cut": unique_cut, "nearest_cos": cos.max(axis=1)}


# ----------------------------------------------------------------------------
# Smart handling: re-seed dead / redundant / killer planes
# ----------------------------------------------------------------------------
def _support_height(n_i, mask, coords, fallback):
    dot = jnp.einsum("dhwc,c->dhw", coords, n_i)
    h = jnp.max(jnp.where(mask, dot, -jnp.inf))
    return jnp.where(jnp.any(mask), h, fallback)


def reseed_planes(key, params, coords, eps, *, score=None,
                  tau_dead=1e-3, tau_dup=5e-2, tau_kill=0.98,
                  tau_par=0.995, margin=1.0, max_reseed=2):
    """
    Move the worst planes to useful places. Call every K steps, outside the
    hot loop (jit it separately).

    score: optional (D,H,W) >= 0 map of "support that should not be there",
           e.g. S * (1 - normalized_density). New planes are then placed
           through voxels sampled from it, facing away from the support
           centroid. Without it they are tangent to the current support in a
           random direction.

    Returns (new_params, reseed_mask). Use reset_plane_opt_state with the mask.
    """
    n = normals_from_params(params)
    d = params["d"]
    N = n.shape[0]
    h = plane_health(params, coords, eps)
    uc = h["unique_cut"]

    killer = uc > tau_kill
    # near-parallel duplicates: j "beats" i if parallel and lower d (tie: index)
    cos = n @ n.T
    idx = jnp.arange(N)
    beats = ((cos > tau_par)
             & ((d[None, :] < d[:, None])
                | ((d[None, :] == d[:, None]) & (idx[None, :] < idx[:, None])))
             & ~jnp.eye(N, dtype=bool))
    dup = jnp.any(beats, axis=1) & (uc < tau_dup)
    # if a killer exists the others only look dead because S ~ 0: leave them
    dead = (uc < tau_dead) & ~jnp.any(killer)
    flag = killer | dead | dup

    # only the worst max_reseed planes per call (killers first)
    priority = jnp.where(killer, -1.0, jnp.where(flag, uc, jnp.inf))
    rank = jnp.argsort(jnp.argsort(priority))
    mask = flag & (rank < max_reseed)

    # shared quantities from the current support
    n_sg, d_sg = lax.stop_gradient((n, d))
    logS = _log_support(n_sg, d_sg, coords, eps)
    S = jnp.exp(logS)
    centroid = (jnp.einsum("dhw,dhwc->c", S, coords)
                / jnp.maximum(jnp.sum(S), 1e-8))
    fallback_d = 0.5 * jnp.max(jnp.abs(coords))
    flat_coords = coords.reshape(-1, coords.shape[-1])
    keys = jax.random.split(key, N)

    def new_plane(args):
        k_i, n_i, d_i, is_killer = args

        def kill_fix(_):
            # keep the normal, pull back to the support of the other planes
            s = _sigma_i(n_i, d_i, coords, eps)
            loo = jnp.exp(logS - jnp.log(jnp.clip(s, _TINY, 1.0)))
            hgt = _support_height(n_i, loo > 0.5, coords, fallback_d)
            return n_i, hgt - margin

        def fresh(_):
            k1, k2 = jax.random.split(k_i)
            if score is not None:
                j = jax.random.categorical(
                    k1, jnp.log(score.reshape(-1) + 1e-12))
                x_star = flat_coords[j]
                v = x_star - centroid
                nn = v / jnp.sqrt(jnp.sum(v ** 2) + 1e-12)
                return nn, jnp.dot(nn, x_star) - margin
            v = jax.random.normal(k1, (3,))
            nn = v / jnp.linalg.norm(v)
            hgt = _support_height(nn, S > 0.5, coords, fallback_d)
            return nn, hgt - margin

        return lax.cond(is_killer, kill_fix, fresh, operand=None)

    def per_plane(args):
        k_i, n_i, d_i, do_it, is_killer = args
        return lax.cond(do_it,
                        new_plane,
                        lambda a: (a[1], a[2]),
                        (k_i, n_i, d_i, is_killer))

    n_new, d_new = lax.map(per_plane, (keys, n, d, mask, killer))
    new_params = _with_normals(params, n_new)
    new_params["d"] = d_new
    return new_params, mask


def reset_plane_opt_state(opt_state, reseed_mask):
    """Zero per-plane optimizer statistics (Adam mu/nu) of reseeded planes."""
    N = reseed_mask.shape[0]

    def f(x):
        if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == N:
            m = reseed_mask.reshape((N,) + (1,) * (x.ndim - 1))
            return jnp.where(m, jnp.zeros_like(x), x)
        return x

    return jax.tree_util.tree_map(f, opt_state)