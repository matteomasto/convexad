# =============================================================================
# OPTIMIZATION
# =============================================================================
import time
from functools import partial
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
import optax

from .losses import compute_Icalc, _center_pad, total_loss
from .model import (init_model, init_params_only, make_coords_for, loss_fn,
                    forward, _amplitude_from)
from .support import make_coords, reseed_planes
from .support_freeform import init_freeform_support_params, invert_support_to_logit


def init_population(key, n_restarts, grid_shape, N=64, size_factor=4.0,
                    phase_type="grid", phase_kwargs=None,
                    support_type="single", support_kwargs=None):
    """Vmapped init of `n_restarts` independent instances.

    Returns
    -------
    params0 : pytree with a leading (n_restarts, ...) axis on every leaf.
    model_static : dict, NOT batched (identical across restarts; computed once
        outside vmap because it contains plain Python strings).
    """
    keys = jax.random.split(key, n_restarts)

    _, model_static = init_model(
        keys[0], grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )
    init_one = partial(
        init_params_only, grid_shape=grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )
    params0 = jax.vmap(init_one)(keys)
    return params0, model_static


def _jvp_via_vjp(f_vjp, y_like, v):
    """J @ v via reverse-mode-only 'vjp of vjp'. Needed because
    halfspace_support only has a custom_vjp -- native jax.jvp/jax.linearize
    raise on it. NOTE: re-validate numerically (finite differences) after any
    change to the backward pass in support.py.
    """
    def h(u):
        return f_vjp(u)[0]
    _, h_vjp = jax.vjp(h, jnp.zeros_like(y_like))
    return h_vjp(v)[0]


def _make_solver(learning_rate, decay_steps, decay_rate, staircase, b1, b2,
                 eps_adam, variant, clip_norm):
    """optax chain: [global norm clip] -> scale_by_<variant> -> decaying lr.

    `learning_rate` may be a traced scalar. The chain's state does not depend
    on its value, so the same optimizer state can be reused after changing it.
    """
    schedule = optax.exponential_decay(
        init_value=learning_rate, transition_steps=decay_steps,
        decay_rate=decay_rate, staircase=staircase,
    )
    if variant == "amsgrad":
        scale = optax.scale_by_amsgrad(b1=b1, b2=b2, eps=eps_adam)
    elif variant == "adabelief":
        scale = optax.scale_by_belief(b1=b1, b2=b2, eps=eps_adam)
    elif variant == "lion":
        scale = optax.scale_by_lion(b1=b1, b2=0.99)
    else:
        raise ValueError(
            f"Unknown variant: {variant!r}, choose 'amsgrad', 'adabelief' or 'lion'.")
    transforms = []
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(clip_norm))
    transforms.append(scale)
    transforms.append(optax.scale_by_learning_rate(schedule))
    return optax.chain(*transforms)


def _advance(params, opt_state, step, stop, static, tol, solver, sign_grad=False):
    """Single instance: take solver steps from (params, opt_state, step) until
    `step == stop` or the gradient norm falls below `tol`. The loss and gradient
    are re-evaluated on entry, so `static` can differ from the previous call.
    Returns (params, opt_state, step, loss at the returned params).
    """
    def f(p):
        return loss_fn(p, static)

    value0, grad0 = jax.value_and_grad(f)(params)

    def cond_fn(carry):
        step, _params, _state, _value, grad = carry
        return jnp.logical_and(step < stop, optax.tree.norm(grad) > tol)

    def body_fn(carry):
        step, params, opt_state, value, grad = carry
        grad_for_update = jax.tree_util.tree_map(jnp.sign, grad) if sign_grad else grad
        direction, opt_state = solver.update(grad_for_update, opt_state, params)
        params = optax.apply_updates(params, direction)
        value, grad = jax.value_and_grad(f)(params)
        return (step + 1, params, opt_state, value, grad)

    step, params, opt_state, value, _grad = lax.while_loop(
        cond_fn, body_fn, (step, params, opt_state, value0, grad0)
    )
    return params, opt_state, step, value


def _solve_one_adam(
    params0, static, max_steps, tol, learning_rate,
    decay_steps=500, decay_rate=0.9, staircase=True,
    b1=0.9, b2=0.98, eps_adam=1e-6,
    variant="amsgrad",
    clip_norm=None,       # global gradient-norm clip, applied before scale_by_*
    sign_grad=False,      # use sign(grad) as the direction fed to scale_by_*
):
    """Single-instance solver with different optimizers."""
    solver = _make_solver(
        learning_rate, decay_steps, decay_rate, staircase, b1, b2, eps_adam,
        variant, clip_norm,
    )
    params, _state, step, value = _advance(
        params0, solver.init(params0), jnp.asarray(0), max_steps, static, tol,
        solver, sign_grad,
    )
    return params, value, step


def residual_fn(params, static):
    """Amplitude-domain residual whose sum-of-squares equals `mse`
    (fidelity term only -- alpha/beta regularizers are not included here).
    """
    support, amplitude, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"],
        static["phase_static"],
        stop_amplitude_grad=static.get("stop_amplitude_grad", False),
        bwd_scale=static.get("bwd_scale", 1.0),
    )
    Iobs = static["Iobs"].astype(jnp.float32)
    Icalc = compute_Icalc(support, amplitude, phase, Iobs)
    denom = jnp.sum(jnp.sqrt(Iobs))
    return (jnp.sqrt(Iobs) - jnp.sqrt(Icalc)).ravel() / jnp.sqrt(denom)


class ReconstructionResult(NamedTuple):
    best_params: dict          # single-instance pytree (argmin over restarts)
    best_loss: jnp.ndarray     # scalar
    all_losses: jnp.ndarray    # (n_restarts,)
    all_steps: jnp.ndarray     # (n_restarts,)
    coords: jnp.ndarray        # (D, H, W, 3), shared
    model_static: dict         # shared; phase_type/support_type/etc.
    eps: float
    all_params: Optional[dict] = None
    opt_state: Optional[Any] = None   # (n_restarts, ...) optimizer state, to resume

    def evaluate(self, Iobs):
        """Recompute (support, amplitude, phase) for the best restart."""
        return forward(self.best_params, self.coords, Iobs, self.eps, self.model_static)


def _freeze(d):
    """Hashable copy of a small static dict (scalar arrays become floats), so
    jax.jit can cache on it."""
    return tuple(sorted(
        (k, float(v) if getattr(v, "shape", None) == () else v) for k, v in d.items()
    ))


@partial(jax.jit, static_argnames=("cfg",))
def _advance_population(params, opt_state, steps, stop, data, hp, cfg):
    """Advance every restart by up to `stop - steps` solver steps.

    Everything that may change between calls (eps, alpha, beta, bwd_scale,
    learning rate, tol, the data, the step limits) is a traced argument, so
    editing it does not recompile; `cfg` holds the structural settings.
    """
    (metric, stop_amplitude_grad, variant, clip_norm, sign_grad, decay_steps,
     decay_rate, staircase, b1, b2, eps_adam, phase_static) = cfg
    static = {
        "coords": data["coords"], "Iobs": data["Iobs"],
        "eps": hp["eps"], "alpha": hp["alpha"], "beta": hp["beta"],
        "bwd_scale": hp["bwd_scale"],
        "metric": metric, "phase_static": dict(phase_static),
        "stop_amplitude_grad": stop_amplitude_grad,
    }
    solver = _make_solver(
        hp["lr"], decay_steps, decay_rate, staircase, b1, b2, eps_adam,
        variant, clip_norm,
    )

    def one(p, s, k, stp):
        return _advance(p, s, k, stp, static, hp["tol"], solver, sign_grad)

    return jax.vmap(one)(params, opt_state, steps, stop)


def _is_ready(x):
    return all(getattr(leaf, "is_ready", lambda: True)()
               for leaf in jax.tree_util.tree_leaves(x))


def _wait(x):
    """Return once every array in `x` has been computed. Polls instead of
    blocking so a KeyboardInterrupt is raised at once, not when the chunk ends."""
    while not _is_ready(x):
        time.sleep(0.005)


# ----------------------------------------------------------------------------
# Plane maintenance between chunks (support_type == "single" only)
# ----------------------------------------------------------------------------
def _cut_score(params, static):
    """(D,H,W) map of support voxels the loss wants removed: relu(dL/dS) * S."""
    support, _, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"],
        static["phase_static"],
    )

    def L(S):
        return total_loss(
            S, _amplitude_from(S, static["Iobs"]), phase, static["Iobs"],
            alpha=static["alpha"], beta=static["beta"], metric=static["metric"],
        )

    return jnp.maximum(jax.grad(L)(support), 0.0) * support


def _make_reseeder(coords, Iobs, alpha, beta, metric, model_static, reseed_kw,
                   use_score):
    """Jitted (key, population params, eps) -> (new params, mask (R, N))."""
    def fn(key, params, eps):
        static = {"coords": coords, "Iobs": Iobs, "eps": eps, "alpha": alpha,
                  "beta": beta, "metric": metric, "phase_static": model_static}
        R = jax.tree_util.tree_leaves(params)[0].shape[0]
        keys = jax.random.split(key, R)
        sup = params["support"]
        if use_score:
            score = jax.vmap(lambda p: _cut_score(p, static))(params)
            new_sup, mask = jax.vmap(
                lambda k, s, sc: reseed_planes(k, s, coords, eps, score=sc, **reseed_kw)
            )(keys, sup, score)
        else:
            new_sup, mask = jax.vmap(
                lambda k, s: reseed_planes(k, s, coords, eps, **reseed_kw)
            )(keys, sup)
        return {**params, "support": new_sup}, mask

    return jax.jit(fn)


def _reset_support_opt_state(opt_state, mask):
    """Zero the optimizer statistics (Adam mu/nu) of reseeded planes.
    `mask` is (n_restarts, N); only leaves under the 'support' key are touched."""
    def f(path, x):
        names = [getattr(p, "key", None) for p in path]
        if ("support" in names and hasattr(x, "ndim") and x.ndim >= 2
                and x.shape[:2] == mask.shape):
            m = mask.reshape(mask.shape + (1,) * (x.ndim - 2))
            return jnp.where(m, jnp.zeros_like(x), x)
        return x

    return jax.tree_util.tree_map_with_path(f, opt_state)


def _renormalize_normals(params):
    """Keep n_raw at unit norm (Adam steps are mostly orthogonal to it, so its
    norm slowly grows and shrinks the effective angular learning rate)."""
    sup = params["support"]
    if isinstance(sup, dict) and "n_raw" in sup:
        v = sup["n_raw"]
        v = v / jnp.sqrt(jnp.sum(v ** 2, axis=-1, keepdims=True) + 1e-12)
        return {**params, "support": {**sup, "n_raw": v}}
    return params


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def reconstruct(
    key, Iobs, n_restarts, N=64, size_factor=4.0, eps=0.6, alpha=0.8, beta=0.1,
    metric="mae", phase_type="grid", phase_kwargs=None, support_type="single",
    support_kwargs=None, max_steps=5000, tol=1e-6, learning_rate=0.05,
    decay_steps=500, decay_rate=0.9, staircase=True, b1=0.9, b2=0.98,
    eps_adam=1e-6, grid_shape=None,
    variant="amsgrad",            # "amsgrad" | "adabelief" | "lion"
    stop_amplitude_grad=False, clip_norm=None, sign_grad=False,
    bwd_scale=1.0,                # >1: wider sigmoid in the half-space backward pass
    reseed_every=None,            # reseed dead/redundant planes every K chunks (None = off)
    reseed_start=500,             # ...but only once the step count reached this
    reseed_use_score=True,        # place new planes using the loss gradient
    reseed_kwargs=None,           # tau_dead, tau_dup, tau_kill, margin, max_reseed, ...
    renormalize=True,             # re-normalize n_raw after every chunk
    resume=None, chunk=100, verbose=True,
):
    """Run `n_restarts` independent optimizations in parallel and keep the best.

    The optimization advances `chunk` steps at a time. Interrupt the kernel (or
    press Ctrl-C) and `reconstruct` returns at once, instead of raising, with
    the result as of the last finished chunk, i.e. the last progress line
    printed. The chunk that was running is dropped. If you interrupt before the
    first chunk has finished, the result is the initial state, with NaN losses.
    Pass it back as `resume=result` to continue from the same parameters,
    optimizer state and step count, with new values of `eps`, `alpha`, `beta`,
    `bwd_scale`, `learning_rate`, `tol` or `max_steps` (a total number of
    steps, not an increment). Changing these does not recompile. Everything
    else must be as in the first call; `key`, `n_restarts` and the model
    arguments are ignored when resuming.

    Plane reseeding (reseed_every) is only available for support_type="single".
    Right after a reseed, the stored losses are those from before it, until the
    next chunk finishes.
    """
    Iobs = jnp.asarray(Iobs, dtype=jnp.float32)

    if resume is None:
        if grid_shape is None:
            grid_shape, coords = make_coords_for(Iobs.shape)
        else:
            coords = make_coords(grid_shape)
        params, model_static = init_population(
            key, n_restarts, grid_shape, N=N, size_factor=size_factor,
            phase_type=phase_type, phase_kwargs=phase_kwargs,
            support_type=support_type, support_kwargs=support_kwargs,
        )
        solver = _make_solver(
            learning_rate, decay_steps, decay_rate, staircase, b1, b2, eps_adam,
            variant, clip_norm,
        )
        opt_state = jax.vmap(solver.init)(params)
        steps = jnp.zeros(n_restarts, dtype=jnp.int32)
    else:
        if resume.opt_state is None:
            raise ValueError("`resume` has no optimizer state; it must come from reconstruct().")
        params, opt_state, steps = resume.all_params, resume.opt_state, resume.all_steps
        coords, model_static = resume.coords, resume.model_static

    cfg = (metric, stop_amplitude_grad, variant, clip_norm, sign_grad, decay_steps,
           decay_rate, staircase, b1, b2, eps_adam, _freeze(model_static))
    hp = {k: float(v) for k, v in
          dict(eps=eps, alpha=alpha, beta=beta, lr=learning_rate, tol=tol,
               bwd_scale=bwd_scale).items()}
    data = {"coords": coords, "Iobs": Iobs}

    is_single = model_static.get("support_type", "single") == "single"
    reseeder = None
    if reseed_every:
        if not is_single:
            raise ValueError("Plane reseeding only supports support_type='single'.")
        reseeder = _make_reseeder(
            coords, Iobs, hp["alpha"], hp["beta"], metric, dict(model_static),
            dict(reseed_kwargs or {}), reseed_use_score,
        )
    n_chunks = 0

    # `carry` is always a complete state: the initial one, replaced (in one
    # assignment) by each finished chunk. `pending` is the chunk being computed.
    carry = (params, opt_state, steps, jnp.full(steps.shape, jnp.nan))
    pending = None
    try:
        while True:
            pending = _advance_population(
                *carry[:3], jnp.minimum(carry[2] + chunk, max_steps), data, hp, cfg,
            )
            _wait(pending)
            moved = bool(jnp.any(pending[2] > carry[2]))
            carry, pending = pending, None
            n_chunks += 1
            if verbose:
                print(f"\rstep {int(carry[2].max())}/{max_steps}   "
                      f"best loss {float(carry[3].min()):.6g}", end="", flush=True)
            if not moved or bool(jnp.all(carry[2] >= max_steps)):
                break

            # ---- between-chunk maintenance (the loop continues after this) ----
            p, s, st, v = carry
            if renormalize and is_single:
                p = _renormalize_normals(p)
            if (reseeder is not None and n_chunks % reseed_every == 0
                    and int(st.max()) >= reseed_start):
                sub = jax.random.fold_in(jax.random.PRNGKey(0), int(st.max()))
                p, mask = reseeder(sub, p, hp["eps"])
                s = _reset_support_opt_state(s, mask)
            carry = (p, s, st, v)
        if verbose:
            print()
    except KeyboardInterrupt:
        if pending is not None and _is_ready(pending):
            carry = pending     # it had just finished; never wait for one that has not
        if verbose:
            print(f"\ninterrupted at step {int(carry[2].max())}: "
                  "pass resume=result to continue")
    params, opt_state, steps, values = carry

    best_idx = jnp.argmin(values)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], params)
    return ReconstructionResult(
        best_params=best_params, best_loss=values[best_idx], all_losses=values,
        all_steps=steps, coords=coords, model_static=model_static, eps=eps,
        all_params=params, opt_state=opt_state,
    )