# =============================================================================
# OPTIMIZATION
# =============================================================================
# Each population member (random restart of support + phase) is an
# independent, deterministic optimization problem sharing only Iobs -- there
# is no minibatch stochasticity here, which is exactly the regime L-BFGS is
# built for. Critically, the population members must NOT share one global
# L-BFGS step: L-BFGS's curvature history mixes information across the whole
# flattened parameter vector, so a single combined solve would spuriously
# correlate unrelated restarts' step sizes and directions. Instead we vmap an
# independent L-BFGS solve over the population axis (in_axes=0 on params,
# in_axes=None on the shared static data/config) and pick the argmin loss.
#
# We use optax.lbfgs (a real limited-memory L-BFGS with a Hessian
# approximation implicit in a short history buffer, not a dense (P, P)
# matrix -- with grid sizes up to ~6.3M voxels a dense BFGS Hessian would be
# infeasible). Peak memory added by the solver's own state is
# ~ 2 * memory_size * (#params) floats, independent of everything except
# the history depth -- this is usually the dominant per-restart memory cost
# at the largest grid sizes, more than the (now O(D*H*W)) support op or the
# FFT buffers. Reduce `memory_size` first if you need to fit more restarts.
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
import optax
# from jax.flatten_util import ravel_pytree
# import jax.scipy.sparse.linalg as jsla

from .losses import compute_Icalc, _center_pad
from .model import init_model, init_params_only, make_coords_for, loss_fn, forward
from .support_freeform import init_freeform_support_params, invert_support_to_logit


def init_population(key, n_restarts, grid_shape, N=64, size_factor=4.0,
                     phase_type="grid", phase_kwargs=None,
                     support_type="single", support_kwargs=None):
    """Vmapped init of `n_restarts` independent instances.

    Returns
    -------
    params0 : pytree with a leading (n_restarts, ...) axis on every leaf.
    model_static : dict, NOT batched (it is identical across restarts by
        construction -- it depends only on grid_shape/phase_type/hkl/
        support_type/etc, never on the random key). Computed once outside
        vmap: it contains plain Python strings (phase_type, support_type)
        that vmap cannot batch.
    """
    keys = jax.random.split(key, n_restarts)

    # Static metadata does not depend on the key -- compute it once, plainly.
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
    raise on it. (A custom_jvp+lax.scan alternative was tried and reverted:
    it broke ordinary jax.grad via a lax.scan-transpose limitation, not
    just added the missing forward-mode path.)
    """
    def h(u):
        return f_vjp(u)[0]
    _, h_vjp = jax.vjp(h, jnp.zeros_like(y_like))
    return h_vjp(v)[0]


# def _newton_step_size(params, grad, direction, static, schedule_alpha, alpha_multiplier=5.0):
#     """Bilinear-Hessian Newton step size (Carlsson et al. 2025, eq. 20):
#     alpha = -<grad, s> / H|params(s, s), s = -direction.

#     H is exact for o -> Icalc -> metric (closed form, one extra FFT); the
#     params -> o layer (support's halfspace_support, amplitude's Parseval
#     normalization, phase's Qnorm*u or phasor) goes through _jvp_via_vjp on
#     the SAME forward() grad already uses -- Qnorm's chain-rule contribution
#     is picked up automatically and exactly, no special-casing needed for
#     phase_type="displacement". See module notes on why this is safe to mix
#     with Qnorm's existing benefit under AMSGrad, and the one real caveat:
#     alpha is a single global scalar over the whole (support, phase)
#     direction, not a per-block step size.

#     metric must be 'mse' or 'poisson' -- matches losses.mse (sqrt/amplitude
#     domain) and losses.poisson_kl (raw intensity domain) exactly, each in
#     its own correct domain. 'mae' is unsupported: h''(I)=0 a.e. for it.

#     Falls back to `schedule_alpha` when H(s,s) <= 0 or the result isn't
#     finite; alpha_max = alpha_multiplier * schedule_alpha (dynamic, tied
#     to the schedule's current value rather than a fixed constant).
#     """
#     metric = static["metric"]
#     if metric not in ("mse", "poisson"):
#         raise ValueError(
#             f"Newton step size only supports metric='mse' or 'poisson' "
#             f"(mae's bilinear Hessian is 0 a.e.), got {metric!r}."
#         )

#     s = jax.tree_util.tree_map(lambda d: -d, direction)

#     def field_fn(p):
#         support, amplitude, phase = forward(
#             p, static["coords"], static["Iobs"], static["eps"],
#             static["phase_static"],
#             stop_amplitude_grad=static.get("stop_amplitude_grad", False),
#         )
#         modulus = support * amplitude
#         if isinstance(phase, tuple):
#             c, sn = phase
#             return jax.lax.complex(modulus * c, modulus * sn)
#         return jax.lax.complex(modulus * jnp.cos(phase), modulus * jnp.sin(phase))

#     o, field_vjp = jax.vjp(field_fn, params)
#     delta_o = _jvp_via_vjp(field_vjp, o, s)

#     Iobs = static["Iobs"].astype(jnp.float32)
#     o_p = _center_pad(o, Iobs.shape)
#     do_p = _center_pad(delta_o, Iobs.shape)

#     z   = jnp.fft.ifftshift(jnp.fft.fftn(jnp.fft.fftshift(o_p)))
#     Fdo = jnp.fft.ifftshift(jnp.fft.fftn(jnp.fft.fftshift(do_p)))

#     Icalc   = jnp.abs(z) ** 2
#     dIcalc  = 2.0 * jnp.real(jnp.conj(z) * Fdo)
#     d2Icalc = 2.0 * jnp.abs(Fdo) ** 2

#     if metric == "mse":
#         D_norm = jnp.sum(jnp.sqrt(Iobs))
#         Icalc_safe = jnp.clip(Icalc, 1e-12, None)
#         sqrtI = jnp.sqrt(Icalc_safe)
#         r = jnp.sqrt(Iobs) - sqrtI
#         h_prime = -r / (D_norm * sqrtI)
#         h_double_prime = jnp.sqrt(Iobs) / (2.0 * D_norm * sqrtI ** 3)
#     else:  # "poisson"
#         N = Iobs.size
#         Icalc_safe = jnp.clip(Icalc, 1e-12, None)
#         h_prime = (1.0 - Iobs / Icalc_safe) / N
#         h_double_prime = (Iobs / Icalc_safe ** 2) / N

#     HH = jnp.sum(h_double_prime * dIcalc ** 2 + h_prime * d2Icalc)

#     grad_dot_s = sum(
#         jnp.sum(g * si) for g, si in zip(
#             jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(s)
#         )
#     )
#     alpha_newton = -grad_dot_s / HH

#     alpha_max = alpha_multiplier * schedule_alpha
#     valid = jnp.logical_and(HH > 0, jnp.isfinite(alpha_newton))
#     return jnp.where(valid, jnp.clip(alpha_newton, 0.0, alpha_max), schedule_alpha)
    
# def _solve_one_adam(
#     params0, static, max_steps, tol, learning_rate,
#     decay_steps=500, decay_rate=0.9, staircase=True,
#     b1=0.9, b2=0.98, eps_adam=1e-6,
#     variant="amsgrad",   # NEW: "amsgrad" | "adabelief" | "lion"
# ):
#     """Single-instance solve with LR decay, run for a fixed number of steps
#     (or until gradient norm drops below `tol`).

#     variant : "amsgrad" (default) | "adabelief" | "lion"
#         All three are optax.GradientTransformations composed with the same
#         exponential-decay LR schedule, so cond_fn/body_fn below are
#         unchanged regardless of variant.
#         - "amsgrad": current default, unchanged.
#         - "adabelief": scales the step by deviation of the gradient from
#           its own EMA ("belief") rather than raw magnitude -- worth trying
#           given the support gradient's clip-boundary masking (active flag
#           in halfspace_support) makes some voxels' gradients intermittently
#           hard-zero, which AdaBelief may register as "low belief" more
#           precisely than AMSGrad's raw-magnitude second moment does.
#         - "lion": sign-of-momentum updates, no second-moment state at all
#           -- every parameter gets the same step magnitude regardless of
#           its raw gradient scale, which is a more forceful answer to the
#           support/amplitude/phase block-scale disparity than any adaptive
#           second-moment method. Needs its own learning_rate tuning: per
#           the optax docs, Lion's suitable LR is typically 3-10x smaller
#           than Adam's for the same problem -- don't reuse the AMSGrad
#           `learning_rate` value unchanged when testing this variant.

#     ** Empirical finding, not just a theoretical concern: ** on this
#     project's actual loss (MAE has an `abs()` kink; the half-space support
#     has a `clip()` kink), a self-consistency test ... [unchanged]
#     """
#     schedule = optax.exponential_decay(
#         init_value=learning_rate,
#         transition_steps=decay_steps,
#         decay_rate=decay_rate,
#         staircase=staircase,
#     )

#     if variant == "amsgrad":
#         scale = optax.scale_by_amsgrad(b1=b1, b2=b2, eps=eps_adam)
#     elif variant == "adabelief":
#         scale = optax.scale_by_belief(b1=b1, b2=b2, eps=eps_adam)
#     elif variant == "lion":
#         scale = optax.scale_by_lion(b1=b1, b2=0.99)  # b2 default per optax; b1 shared with caller
#     else:
#         raise ValueError(f"Unknown variant: {variant!r}, choose 'amsgrad', 'adabelief' or 'lion'.")

#     solver = optax.chain(scale, optax.scale_by_learning_rate(schedule))

#     def f(p):
#         return loss_fn(p, static)

#     opt_state0 = solver.init(params0)
#     value0, grad0 = jax.value_and_grad(f)(params0)

#     def cond_fn(carry):
#         step, _params, _state, _value, grad = carry
#         gnorm = optax.tree.norm(grad)
#         return jnp.logical_and(step < max_steps, gnorm > tol)

#     def body_fn(carry):
#         step, params, opt_state, value, grad = carry
#         updates, opt_state = solver.update(grad, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         value, grad = jax.value_and_grad(f)(params)
#         return (step + 1, params, opt_state, value, grad)

#     init_carry = (jnp.asarray(0), params0, opt_state0, value0, grad0)
#     final_step, final_params, _final_state, final_value, _final_grad = lax.while_loop(
#         cond_fn, body_fn, init_carry
#     )
#     return final_params, final_value, final_step

def _solve_one_adam(
    params0, static, max_steps, tol, learning_rate,
    decay_steps=500, decay_rate=0.9, staircase=True,
    b1=0.9, b2=0.98, eps_adam=1e-6,
    variant="amsgrad",
    clip_norm=None,       # global gradient-norm clip, applied before scale_by_*
    sign_grad=False,      # use sign(grad) as the direction fed to scale_by_*
):
    """... existing docstring ...

    clip_norm : float or None
        If set, clips the global L2 norm of the raw gradient to this value
        before AMSGrad/AdaBelief/Lion normalization -- a bounded-magnitude
        gradient regardless of how wrong the current point is, testing
        whether `mae`+Adam's advantage on this landscape is really "bounded
        step size" rather than anything MAE-specific. Composes with any
        `variant`.
    sign_grad : bool
        If True, replaces the gradient with elementwise sign(grad) before
        it reaches scale_by_*, i.e. every voxel/parameter contributes a
        fixed-magnitude push in its gradient's direction only -- a closer
        structural match to mae's own gradient (+-1/sum(Iobs) per voxel,
        magnitude-independent of the residual) than clip_norm is. Mutually
        compatible with clip_norm (sign first would make clipping a
        no-op on magnitude; apply clip_norm to the raw gradient, sign_grad
        replaces it entirely -- if both are set, sign_grad wins, since
        clipping a vector of +-1's does nothing meaningful).
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
        raise ValueError(f"Unknown variant: {variant!r}, choose 'amsgrad', 'adabelief' or 'lion'.")

    transforms = []
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(clip_norm))
    transforms.append(scale)
    transforms.append(optax.scale_by_learning_rate(schedule))
    solver = optax.chain(*transforms)

    def f(p):
        return loss_fn(p, static)

    opt_state0 = solver.init(params0)
    value0, grad0 = jax.value_and_grad(f)(params0)

    def cond_fn(carry):
        step, _params, _state, _value, grad = carry
        return jnp.logical_and(step < max_steps, optax.tree.norm(grad) > tol)

    def body_fn(carry):
        step, params, opt_state, value, grad = carry

        grad_for_update = (
            jax.tree_util.tree_map(jnp.sign, grad) if sign_grad else grad
        )
        direction, opt_state = solver.update(grad_for_update, opt_state, params)

        params = optax.apply_updates(params, direction)   # was: `updates` (undefined)
        value, grad = jax.value_and_grad(f)(params)
        return (step + 1, params, opt_state, value, grad)

    init_carry = (jnp.asarray(0), params0, opt_state0, value0, grad0)
    final_step, final_params, _final_state, final_value, _final_grad = lax.while_loop(
        cond_fn, body_fn, init_carry
    )
    return final_params, final_value, final_step
    
def residual_fn(params, static):
    """Amplitude-domain residual whose sum-of-squares equals `mse`
    (fidelity term only -- alpha/beta regularizers are not included here).
    """
    support, amplitude, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"],
        static["phase_static"],
        stop_amplitude_grad=static.get("stop_amplitude_grad", False),
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

    def evaluate(self, Iobs):
        """Recompute (support, amplitude, phase) for the best restart."""
        return forward(self.best_params, self.coords, Iobs, self.eps, self.model_static)


def reconstruct(
    key,
    Iobs,
    n_restarts,
    N=64,
    size_factor=4.0,
    eps=0.6,
    alpha=0.8,
    beta=0.1,
    metric="mae",
    phase_type="grid",
    phase_kwargs=None,
    support_type="single",
    support_kwargs=None,
    max_steps=5000,
    tol=1e-6,
    learning_rate=0.05,
    decay_steps=500,
    decay_rate=0.9,
    staircase=True,
    b1=0.9,
    b2=0.98,
    eps_adam=1e-6,
    grid_shape=None,
    variant="amsgrad",            # "amsgrad" | "adabelief" | "lion"
    stop_amplitude_grad=False,    # restored
    clip_norm=None,     # NEW
    sign_grad=False,    # NEW
):
    Iobs = jnp.asarray(Iobs, dtype=jnp.float32)
    if grid_shape is None:
        grid_shape, coords = make_coords_for(Iobs.shape)
    else:
        from .support import make_coords
        coords = make_coords(grid_shape)

    params0, model_static = init_population(
        key, n_restarts, grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )

    static = {
        "coords": coords,
        "Iobs": Iobs,
        "eps": eps,
        "alpha": alpha,
        "beta": beta,
        "metric": metric,
        "phase_static": model_static,
        "stop_amplitude_grad": stop_amplitude_grad,
    }

    solve = partial(
        _solve_one_adam, static=static, max_steps=max_steps, tol=tol,
        learning_rate=learning_rate, decay_steps=decay_steps,
        decay_rate=decay_rate, staircase=staircase,
        b1=b1, b2=b2, eps_adam=eps_adam, variant=variant,
        clip_norm=clip_norm, sign_grad=sign_grad,
    )
    batched_solve = jax.vmap(solve, in_axes=(0,))

    final_params, final_values, final_steps = batched_solve(params0)

    best_idx = jnp.argmin(final_values)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], final_params)

    return ReconstructionResult(
        best_params=best_params,
        best_loss=final_values[best_idx],
        all_losses=final_values,
        all_steps=final_steps,
        coords=coords,
        model_static=model_static,
        eps=eps,
        all_params=final_params,
    )


# =============================================================================
# TWO-STAGE (CONVEX -> FREE-FORM) RECONSTRUCTION
# =============================================================================
# Stage 1 is a normal `reconstruct(..., support_type="single" | "multi")`
# population solve: cheap (O(N) or O(M*N) support params/restart), and its
# job is only to find the rough global shape and phase field, robustly,
# via the usual multi-restart search.
#
# Stage 2 releases a SMALL number of the best stage-1 restarts to the
# free-form (support_freeform.py) voxel-grid support and refines them,
# each anchored to its own stage-1 shape via a T (sharpness) / zeta
# (anchor weight) continuation schedule that starts tight (trust the
# stage-1 shape) and relaxes (let genuinely non-convex detail -- a twin
# facet, a second particle, a concave notch -- emerge). See
# support_freeform.py's module docstring for why a cold free-form start is
# a bad idea.

def _solve_one_adam_freeform(
    params0, S_ref, base_static, stage_schedule, tol,
    learning_rate, decay_steps, decay_rate, staircase, b1, b2, eps_adam,
):
    """Single-instance staged Adam solve for the free-form support.

    Unlike `_solve_one_adam`, `S_ref` (this restart's anchor target -- the
    stage-1 support it was warm-started from) is an explicit argument
    rather than folded into `base_static`, specifically so it can vary per
    population member under `vmap(..., in_axes=(0, 0))` while
    `base_static` (Iobs, coords, alpha, beta, gamma, delta, metric,
    phase_static) stays shared (`in_axes=None`). This is what lets several
    kept stage-1 restarts be released to free-form in parallel, each
    anchored to its own stage-1 solution, instead of all sharing one.

    `stage_schedule` is a small Python-level (static, not traced) sequence
    of (T, zeta, max_steps) triples -- a continuation schedule for the
    support sharpness T (passed through as `static["eps"]`, reusing the
    existing eps-as-softness convention from support.py/multi_support.py)
    and the anchor weight zeta. Each stage is one `_solve_one_adam`-style
    `lax.while_loop`; the stages are unrolled in plain Python (there are
    only ever a handful), so this still traces to a single jaxpr per
    restart under vmap/jit, and gets a fresh Adam state at each stage
    boundary (a small, deliberate reset -- the loss landscape genuinely
    changes shape each time T or zeta changes, so stale second-moment
    estimates from the previous stage aren't worth carrying over).
    """
    params = params0
    final_value = jnp.asarray(jnp.inf, dtype=jnp.float32)
    total_steps = jnp.asarray(0)

    for T_i, zeta_i, steps_i in stage_schedule:
        stage_static = dict(base_static)
        stage_static["eps"] = T_i
        stage_static["zeta"] = zeta_i
        stage_static["S_ref"] = S_ref
        params, final_value, steps = _solve_one_adam(
            params, stage_static, max_steps=steps_i, tol=tol,
            learning_rate=learning_rate, decay_steps=decay_steps,
            decay_rate=decay_rate, staircase=staircase,
            b1=b1, b2=b2, eps_adam=eps_adam,
        )
        total_steps = total_steps + steps

    return params, final_value, total_steps


def reconstruct_two_stage(
    key,
    Iobs,
    n_restarts_stage1,
    stage1_support_type="single",
    n_keep=1,
    n_restarts_stage2=None,
    grid_shape=None,
    stage1_kwargs=None,
    T_schedule=(1.5, 0.6, 0.25),
    zeta_schedule=(3.0, 1.0, 0.0),
    steps_per_stage=1000,
    gamma=1e-3,
    delta=1e-2,
    alpha2=None,
    beta2=None,
    noise_scale_support=0.0,
    noise_scale_phase=0.0,
    stage2_learning_rate=0.02,
    stage2_decay_steps=500,
    stage2_decay_rate=0.9,
    stage2_staircase=True,
    stage2_b1=0.9,
    stage2_b2=0.98,
    stage2_eps_adam=1e-6,
    tol=1e-6,
):
    """Two-stage reconstruction for non-convex / multi-particle supports:
    a cheap convex/multi-convex stage 1, then a free-form release for a
    small population of the best stage-1 candidates.

    Parameters
    ----------
    n_restarts_stage1, stage1_support_type : population size and support
        parameterization ("single" or "multi") for the stage-1 solve --
        forwarded to `reconstruct` via `stage1_kwargs`.
    n_keep : how many of the lowest-loss stage-1 restarts to carry into
        stage 2. Keep this small (1-3): each one becomes an O(D*H*W)-param
        free-form restart, a very different memory regime than stage 1
        (see `support_freeform.py`'s and `support.py`'s module docstrings
        on per-restart memory cost).
    n_restarts_stage2 : population size for stage 2; defaults to `n_keep`
        (one free-form restart per kept candidate). If larger, extra
        restarts are assigned to kept candidates round-robin and
        decorrelated via `noise_scale_support`/`noise_scale_phase`.
    stage1_kwargs : dict forwarded to `reconstruct` for stage 1 (N,
        size_factor, eps, alpha, beta, metric, phase_type, phase_kwargs,
        support_kwargs, optimizer, max_steps, tol, memory_size,
        learning_rate, ...). `support_type` is set from
        `stage1_support_type` if not already present.
    T_schedule, zeta_schedule : continuation schedule for stage 2 -- same
        length, T decreasing (support sharpness, see
        `support_freeform.compute_freeform_support`) and zeta decreasing
        toward 0 (anchor weight toward the stage-1 shape, see
        `support_freeform.anchor_penalty`). Defaults are a reasonable
        starting point, not tuned for any particular dataset.
    steps_per_stage : Adam steps (or until `tol`) per schedule stage.
    gamma, delta : perimeter / double-well regularizer weights (see
        `support_freeform.tv_support` / `double_well_support`), held
        constant across all stage-2 stages. These do NOT resist splitting
        into disconnected domains or growing concavities -- both cost the
        same perimeter per unit boundary area as a convex bulge -- so
        setting them to 0 to "give the field more freedom" is usually
        counterproductive: it removes the pressure that keeps voxels
        decisively in/out, which is what makes a genuine gap between two
        domains (or a genuine notch) show up as S~0 instead of a blurry
        S~0.4 compromise.
    alpha2, beta2 : support-size / phase-TV weights for stage 2, DEFAULT
        None (reuse stage 1's `alpha`/`beta` from `stage1_kwargs`, for
        backwards compatibility). Pass these explicitly whenever stage 1
        used `alpha=0` (common and often correct for the convex/multi-
        convex parameterization, which cannot represent diffuse background
        mass at all) -- the free-form support has no such structural
        protection: every voxel is an independent parameter, so with
        alpha=0 (and gamma=delta=0) nothing prices total support mass in
        stage 2, and as T softens, leaked S from the (typically much
        larger) background volume can dominate `amplitude =
        sqrt(sum_I / (N*sum_S))` and destabilize the fit. A small nonzero
        alpha2 (and/or gamma, delta) is usually needed even when stage 1's
        alpha is legitimately 0.
    noise_scale_support, noise_scale_phase : stddev of Gaussian noise added
        to the support logit / phase params of each stage-2 restart on top
        of its assigned stage-1 warm start -- only matters when
        `n_restarts_stage2 > n_keep`, to decorrelate restarts that would
        otherwise be exact duplicates.

    Returns
    -------
    stage1_result, stage2_result : both `ReconstructionResult`. On
    `stage2_result`, `eps` is set to `T_schedule[-1]` (the T needed to
    reproduce the final support via `.evaluate()`), and `model_static`
    has `support_type` overridden to `"freeform"`.
    """
    if len(T_schedule) != len(zeta_schedule):
        raise ValueError("T_schedule and zeta_schedule must have the same length")

    key1, key2 = jax.random.split(key)
    stage1_kwargs = dict(stage1_kwargs or {})
    stage1_kwargs.setdefault("support_type", stage1_support_type)
    alpha = stage1_kwargs.get("alpha", 0.8)
    beta = stage1_kwargs.get("beta", 0.1)
    metric = stage1_kwargs.get("metric", "mae")
    alpha_stage2 = alpha if alpha2 is None else alpha2
    beta_stage2 = beta if beta2 is None else beta2

    stage1_result = reconstruct(
        key1, Iobs, n_restarts=n_restarts_stage1, grid_shape=grid_shape,
        **stage1_kwargs,
    )

    Iobs_arr = jnp.asarray(Iobs, dtype=jnp.float32)
    coords = stage1_result.coords
    grid_shape = coords.shape[:3]

    n_keep = min(n_keep, n_restarts_stage1)
    keep_idx = jnp.argsort(stage1_result.all_losses)[:n_keep]
    kept_params = jax.tree_util.tree_map(lambda x: x[keep_idx], stage1_result.all_params)

    # Re-evaluate the kept restarts' converged support at stage 1's own
    # (fixed) eps -- this is the shape stage 2 warm-starts from and
    # anchors to.
    kept_forward = jax.vmap(
        lambda p: forward(p, coords, Iobs_arr, stage1_result.eps, stage1_result.model_static)
    )
    kept_support, _kept_amplitude, _kept_phase = kept_forward(kept_params)  # (n_keep, D,H,W)
    kept_logit = jax.vmap(invert_support_to_logit)(kept_support)            # (n_keep, D,H,W)

    n_restarts_stage2 = n_restarts_stage2 or n_keep
    assign = jnp.arange(n_restarts_stage2) % n_keep  # round-robin over kept candidates
    logits0 = kept_logit[assign]
    phase0 = jax.tree_util.tree_map(lambda x: x[assign], kept_params["phase"])
    S_ref_batched = kept_support[assign]                                    # (n_restarts_stage2, D,H,W)

    def _init_stage2_one(k, logit0, phase_p0):
        k_support, k_phase = jax.random.split(k)
        support_params = init_freeform_support_params(
            k_support, grid_shape, init_logit=logit0, noise_scale=noise_scale_support
        )
        if noise_scale_phase > 0.0:
            phase_params = jax.tree_util.tree_map(
                lambda x: x + noise_scale_phase * jax.random.normal(k_phase, x.shape),
                phase_p0,
            )
        else:
            phase_params = phase_p0
        return {"support": support_params, "phase": phase_params}

    keys2 = jax.random.split(key2, n_restarts_stage2)
    params0_stage2 = jax.vmap(_init_stage2_one)(keys2, logits0, phase0)

    model_static_stage2 = dict(stage1_result.model_static)
    model_static_stage2["support_type"] = "freeform"

    base_static = {
        "coords": coords,
        "Iobs": Iobs_arr,
        "alpha": alpha_stage2,
        "beta": beta_stage2,
        "metric": metric,
        "gamma": gamma,
        "delta": delta,
        "phase_static": model_static_stage2,
    }
    stage_schedule = tuple(
        (T_i, zeta_i, steps_per_stage) for T_i, zeta_i in zip(T_schedule, zeta_schedule)
    )

    solve = partial(
        _solve_one_adam_freeform, base_static=base_static, stage_schedule=stage_schedule,
        tol=tol, learning_rate=stage2_learning_rate, decay_steps=stage2_decay_steps,
        decay_rate=stage2_decay_rate, staircase=stage2_staircase,
        b1=stage2_b1, b2=stage2_b2, eps_adam=stage2_eps_adam,
    )
    batched_solve = jax.vmap(solve, in_axes=(0, 0))
    final_params, final_values, final_steps = batched_solve(params0_stage2, S_ref_batched)

    best_idx = jnp.argmin(final_values)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], final_params)

    stage2_result = ReconstructionResult(
        best_params=best_params,
        best_loss=final_values[best_idx],
        all_losses=final_values,
        all_steps=final_steps,
        coords=coords,
        model_static=model_static_stage2,
        eps=T_schedule[-1],
        all_params=final_params,
    )
    return stage1_result, stage2_result
