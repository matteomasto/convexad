"""Half-space (convex polytope) support for ConvexAD, with dead-plane handling.

Numerics, init and projection:
  * forward uses log_sigmoid, backward uses sigmoid(-t): exact, no float32 hard zero
  * planes are parametrised by an unnormalised vector v (n = v/|v|): no pole singularity
  * no gradient w.r.t. coords (constant input)
  * Fibonacci-spread, randomly rotated initial normals with jittered offsets
  * tighten(): one-sided projection pulling planes outside the support back to tangency

Diagnostics and replacement:
  * plane_activity(): cheap share of the support carried by each plane
  * plane_health(): leave-one-out "unique cut" per plane -> dead / shadowed / killer
  * reseed_planes(): re-place dead, duplicate and killer planes (optionally guided by a
    loss-derived score map)
"""

import jax
import jax.numpy as jnp
from jax import lax


# --------------------------------------------------------------------------- #
# Coordinates and helpers
# --------------------------------------------------------------------------- #
def make_coords(grid_shape):
    D, H, W = grid_shape
    z = jnp.linspace(-(D - 1) / 2.0, (D - 1) / 2.0, D)
    y = jnp.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, H)
    x = jnp.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, W)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    return jnp.stack([xx, yy, zz], axis=-1)  # (D, H, W, 3)


def _t(n_i, d_i, coords, eps):
    """Signed distance to plane i in units of eps; > 0 means inside the half-space."""
    return (d_i - jnp.einsum("dhwc,c->dhw", coords, n_i)) / eps


def to_unit(v):
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _log_support(n, d, coords, eps):
    """log S = sum_i log sigmoid(t_i), O(DHW) memory."""
    def step(logS, nd_i):
        n_i, d_i = nd_i
        return logS + jax.nn.log_sigmoid(_t(n_i, d_i, coords, eps)), None

    logS, _ = lax.scan(step, jnp.zeros(coords.shape[:3], coords.dtype), (n, d))
    return logS


# --------------------------------------------------------------------------- #
# Soft polytope indicator with memory-light custom gradient
# --------------------------------------------------------------------------- #
@jax.custom_vjp
def halfspace_support(n, d, coords, eps):
    """S(x) = prod_i sigmoid((d_i - n_i.x)/eps). Memory O(D*H*W), not O(D*H*W*N)."""
    return jnp.exp(_log_support(n, d, coords, eps))


def _fwd(n, d, coords, eps):
    S = jnp.exp(_log_support(n, d, coords, eps))
    return S, (n, d, coords, eps, S)


def _bwd(res, g):
    n, d, coords, eps, S = res
    gS = g * S
    flat_coords = coords.reshape(-1, coords.shape[-1])

    def per_plane(nd_i):
        n_i, d_i = nd_i
        # (1 - sigma)/eps written as sigmoid(-t)/eps: no cancellation, representable
        # down to ~e^-87 in float32 ((1 - sigma) is exactly 0 beyond t ~ 17).
        w = gS * jax.nn.sigmoid(-_t(n_i, d_i, coords, eps)) / eps
        return -(w.reshape(-1) @ flat_coords), jnp.sum(w)

    dn, dd = lax.map(per_plane, (n, d))   # sequential: O(DHW) memory
    return dn, dd, None, None             # coords and eps are constants


halfspace_support.defvjp(_fwd, _bwd)


def compute_support(params, coords, eps):
    return halfspace_support(to_unit(params["v"]), params["d"], coords, eps)


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def fibonacci_sphere(N):
    i = jnp.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    r = jnp.sqrt(1.0 - z ** 2)
    phi = jnp.pi * (1.0 + 5.0 ** 0.5) * i
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


def init_support_params(key, N, grid_shape, size_factor=4.0, d_jitter=0.05):
    """Evenly spread normals, random rotation per instance, jittered radii."""
    key_q, key_d = jax.random.split(key)
    Q, Rm = jnp.linalg.qr(jax.random.normal(key_q, (3, 3)))
    Q = Q * jnp.sign(jnp.diag(Rm))                      # uniform random rotation/reflection
    v0 = fibonacci_sphere(N) @ Q.T                      # (N, 3), unit norm
    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    d0 = R * (1.0 + d_jitter * jax.random.normal(key_d, (N,)))
    return {"v": v0, "d": d0}


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def plane_activity(params, coords, eps):
    """Share of the support carried by each plane, shape (N,). Cheap.

    ~0 means the plane neither cuts nor touches the support (it is dead).
    Cannot tell duplicates apart or detect killers; use plane_health for that.
    """
    params = lax.stop_gradient(params)
    n = to_unit(params["v"])
    S = compute_support(params, coords, eps)

    def act(nd):
        n_i, d_i = nd
        return jnp.sum(S * jax.nn.sigmoid(-_t(n_i, d_i, coords, eps)))

    return lax.map(act, (n, params["d"])) / (jnp.sum(S) + 1e-12)


def plane_health(params, coords, eps):
    """Leave-one-out diagnostics.

    unique_cut[i] in [0, 1]: fraction of the volume that would exist WITHOUT plane i
        which plane i alone removes.
          ~0 -> dead (outside the object) or shadowed by other planes
          ~1 -> killer (removes almost everything the others leave)
    nearest_cos[i]: max cosine similarity with any other plane (redundancy).
    """
    params = lax.stop_gradient(params)
    n = to_unit(params["v"])
    d = params["d"]
    logS = _log_support(n, d, coords, eps)

    def per_plane(nd_i):
        n_i, d_i = nd_i
        t = _t(n_i, d_i, coords, eps)
        loo = jnp.exp(logS - jax.nn.log_sigmoid(t))      # S without plane i
        return jnp.sum(loo * jax.nn.sigmoid(-t)), jnp.sum(loo)

    excl, vol_wo = lax.map(per_plane, (n, d))
    unique_cut = excl / jnp.maximum(vol_wo, 1e-8)

    N = n.shape[0]
    cos = jnp.where(jnp.eye(N, dtype=bool), -1.0, n @ n.T)
    return {"vol": jnp.sum(jnp.exp(logS)), "unique_cut": unique_cut,
            "nearest_cos": cos.max(axis=1)}


# --------------------------------------------------------------------------- #
# Projection: tighten
# --------------------------------------------------------------------------- #
def _support_height(n_i, mask, coords, fallback):
    dot = jnp.einsum("dhwc,c->dhw", coords, n_i)
    h = jnp.max(jnp.where(mask, dot, -jnp.inf))
    return jnp.where(jnp.any(mask), h, fallback)


def tighten(params, coords, eps, margin=2.0, thresh=0.5, slack=3.0):
    """One-sided projection: pull planes lying far outside the support back to tangency.

    With h_i the support function of {S > thresh} along n_i (so h_i <= d_i):
        d_i <- h_i + margin*eps   only if   d_i > h_i + slack,   else unchanged.
    Planes are never pushed outward, and planes within `slack` voxels of the support
    (binding planes: the voxel-discretisation gap and the receding level set at
    vertices are typically ~1-2 voxels) are never touched. That hysteresis is what keeps
    tighten from fighting the optimiser near convergence: only genuinely far-away
    (dead / shadowed) planes move. margin*eps keeps a live gradient at the tangent
    point. Also renormalises v.
    """
    params = lax.stop_gradient(params)
    n = to_unit(params["v"])
    inside = compute_support(params, coords, eps) > thresh
    any_inside = jnp.any(inside)
    h = lax.map(lambda n_i: _support_height(n_i, inside, coords, -jnp.inf), n)
    far = params["d"] > h + slack
    d_new = jnp.where(any_inside & far, h + margin * eps, params["d"])
    return {"v": n, "d": d_new}


# --------------------------------------------------------------------------- #
# Re-seeding of dead / duplicate / killer planes
# --------------------------------------------------------------------------- #
def reseed_planes(key, params, coords, eps, *, score=None,
                  tau_dead=1e-3, tau_dup=5e-2, tau_kill=0.98, tau_par=0.995,
                  margin=2.0, max_reseed=2):
    """Move the worst planes to useful places. Call every K steps, outside the hot loop.

    flagged planes:
      killer    unique_cut > tau_kill        -> keep normal, pull back to the support
                                                of the other planes (+ margin*eps)
      dead      unique_cut < tau_dead        -> fresh plane (skipped if a killer exists:
                                                then the others only look dead)
      duplicate near-parallel (cos > tau_par) to a plane that "beats" it (lower d, or
                lower index on ties) and unique_cut < tau_dup -> fresh plane
    Only the `max_reseed` worst planes are moved per call (killers first).

    fresh plane: always tangent to the current support (+ margin*eps), so it starts
    alive without changing the support; the optimizer then decides how far to cut.
      score given   direction from the support centroid towards a voxel sampled from
                    `score` (>= 0, (D,H,W), e.g. relu(dLoss/dS) * S); falls back to a
                    random direction if that voxel coincides with the centroid
      no score      random direction

    Returns (new_params, reseed_mask).
    """
    params = lax.stop_gradient(params)
    n = to_unit(params["v"])
    d = params["d"]
    N = n.shape[0]
    uc = plane_health(params, coords, eps)["unique_cut"]

    killer = uc > tau_kill
    cos = n @ n.T
    idx = jnp.arange(N)
    beats = ((cos > tau_par)
             & ((d[None, :] < d[:, None])
                | ((d[None, :] == d[:, None]) & (idx[None, :] < idx[:, None])))
             & ~jnp.eye(N, dtype=bool))
    dup = jnp.any(beats, axis=1) & (uc < tau_dup)
    dead = (uc < tau_dead) & ~jnp.any(killer)
    flag = killer | dead | dup

    priority = jnp.where(killer, -1.0, jnp.where(flag, uc, jnp.inf))
    rank = jnp.argsort(jnp.argsort(priority))
    mask = flag & (rank < max_reseed)

    logS = _log_support(n, d, coords, eps)
    S = jnp.exp(logS)
    centroid = (jnp.einsum("dhw,dhwc->c", S, coords)
                / jnp.maximum(jnp.sum(S), 1e-8))
    # = 0.5 * max|coords|, from the static grid shape (no reduction over the array)
    fallback_d = 0.25 * (max(coords.shape[:3]) - 1)
    flat_coords = coords.reshape(-1, coords.shape[-1])
    keys = jax.random.split(key, N)

    def new_plane(args):
        k_i, n_i, d_i, is_killer = args

        def kill_fix(_):
            loo = jnp.exp(logS - jax.nn.log_sigmoid(_t(n_i, d_i, coords, eps)))
            hgt = _support_height(n_i, loo > 0.5, coords, fallback_d)
            return n_i, hgt + margin * eps

        def fresh(_):
            k1, k2 = jax.random.split(k_i)
            v = jax.random.normal(k2, (3,))                     # random fallback direction
            if score is not None:
                j = jax.random.categorical(k1, jnp.log(score.reshape(-1) + 1e-12))
                u = flat_coords[j] - centroid
                v = jnp.where(jnp.linalg.norm(u) > 1e-3, u, v)  # guard degenerate case
            nn = v / (jnp.linalg.norm(v) + 1e-12)
            hgt = _support_height(nn, S > 0.5, coords, fallback_d)
            return nn, hgt + margin * eps

        return lax.cond(is_killer, kill_fix, fresh, jnp.zeros(()))

    def per_plane(args):
        k_i, n_i, d_i, do_it, is_killer = args
        return lax.cond(do_it, new_plane, lambda a: (a[1], a[2]),
                        (k_i, n_i, d_i, is_killer))

    n_new, d_new = lax.map(per_plane, (keys, n, d, mask, killer))
    return {"v": n_new, "d": d_new}, mask


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    grid = (24, 24, 24)
    N, eps = 12, 0.6
    coords = make_coords(grid)
    params = init_support_params(jax.random.PRNGKey(0), N, grid)

    # 1) custom gradient vs plain autodiff reference
    def ref_support(n, d):
        logS = sum(jax.nn.log_sigmoid(_t(n[i], d[i], coords, eps)) for i in range(N))
        return jnp.exp(logS)

    wgt = jax.random.normal(jax.random.PRNGKey(1), grid)
    f_custom = lambda p: jnp.sum(wgt * compute_support(p, coords, eps))
    f_ref = lambda p: jnp.sum(wgt * ref_support(to_unit(p["v"]), p["d"]))
    g1, g2 = jax.grad(f_custom)(params), jax.grad(f_ref)(params)
    for k in ("v", "d"):
        err = jnp.max(jnp.abs(g1[k] - g2[k])) / (jnp.max(jnp.abs(g2[k])) + 1e-12)
        print(f"grad {k}: max relative error = {float(err):.2e}")

    # 2) dead planes: push half of them far outside, check tighten
    dead_p = {"v": params["v"], "d": params["d"].at[: N // 2].set(40.0)}
    act0 = plane_activity(dead_p, coords, eps)
    tight = tighten(dead_p, coords, eps)
    act1 = plane_activity(tight, coords, eps)
    dS = jnp.max(jnp.abs(compute_support(tight, coords, eps)
                         - compute_support(dead_p, coords, eps)))
    print("activity before tighten (dead half):", [f"{a:.1e}" for a in act0[: N // 2]])
    print("activity after  tighten (same planes):", [f"{a:.1e}" for a in act1[: N // 2]])
    print(f"max |S_after - S_before| = {float(dS):.2e}")

    # 3) health + reseed: duplicate a plane, add a killer, check detection and repair
    pd = {"v": params["v"].at[1].set(params["v"][0]), "d": params["d"].at[1].set(params["d"][0] + 0.3)}
    print("unique_cut with a duplicate:", [f"{u:.2e}" for u in plane_health(pd, coords, eps)["unique_cut"]])
    pk = {"v": params["v"], "d": params["d"].at[2].set(-30.0)}
    print("killer unique_cut:", float(plane_health(pk, coords, eps)["unique_cut"][2]))
    fixed, mask = reseed_planes(jax.random.PRNGKey(3), pk, coords, eps)
    print("reseeded planes:", jnp.nonzero(mask)[0].tolist(),
          "| support volume after:", float(compute_support(fixed, coords, eps).sum()))