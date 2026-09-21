# =============================================================================
# MODEL
# =============================================================================
# Functional, single-instance model: `params` is a plain dict pytree (JAX
# treats dicts as pytrees natively -- no custom registration needed), and
# everything that does not need a gradient (Iobs, coords, eps, loss weights,
# phase_type, ...) lives in a separate `static` dict. `optimize.py` vmaps
# these functions over a leading population axis of `params`, with `static`
# passed as an unbatched (in_axes=None) argument.
import jax
import jax.numpy as jnp

from .support import compute_support, init_support_params, make_coords
from .multi_support import init_multi_support_params, compute_multi_support
from .support_freeform import init_freeform_support_params, compute_freeform_support
from .phase import compute_phase, compute_phasor, init_phase_params
from .losses import total_loss


def make_coords_for(Iobs_shape):
    """Convenience: derive grid_shape from an observed-intensity shape the
    same way the original implementation did (grid = Iobs // 2), and build
    the fixed coordinate grid for it.
    """
    D, H, W = Iobs_shape
    grid_shape = (D // 2, H // 2, W // 2)
    return grid_shape, make_coords(grid_shape)


def init_model(key, grid_shape, N=64, size_factor=4.0, phase_type="grid",
               phase_kwargs=None, support_type="single", support_kwargs=None):
    """Initialize a single instance's trainable parameters.

    Parameters
    ----------
    support_type : "single" (default) | "multi" | "freeform"
        "single": one convex object, `N` half-spaces (as before).
        "multi": a non-convex object built as a soft union of several
            convex parts -- see multi_support.py. `support_kwargs` may
            contain "M_parts" (default 2), "N_per_part" (default N),
            "use_gates" (default False, learnable per-part on/off gates).
        "freeform": one free logit per voxel -- see support_freeform.py.
            Intended as a stage-2 release from a converged "single" or
            "multi" solution, not a cold start (see
            `optimize.reconstruct_two_stage`). `support_kwargs` may
            contain "init_logit" ((D,H,W) array, typically
            `support_freeform.invert_support_to_logit(S_stage1)`) and
            "noise_scale" (default 0.0, to decorrelate a population of
            restarts sharing the same warm start).

    Returns
    -------
    params : dict pytree of trainable arrays.
    model_static : dict of non-trainable metadata: "phase_type" (and, for
        'displacement', "Qnorm"), plus "support_type" (and, for "multi",
        "use_gates"). This dict is threaded through `forward`/`loss_fn`
        exactly where `phase_static` used to be -- existing code that only
        reads `model_static["phase_type"]` etc. is unaffected, since keys
        are only ever added here, never renamed or removed.
    """
    key_support, key_phase = jax.random.split(key)
    support_kwargs = support_kwargs or {}

    if support_type == "single":
        support_params = init_support_params(key_support, N, grid_shape, size_factor)
        support_static = {"support_type": "single"}
    elif support_type == "multi":
        M_parts = support_kwargs.get("M_parts", 2)
        N_per_part = support_kwargs.get("N_per_part", N)
        use_gates = support_kwargs.get("use_gates", False)
        support_params = init_multi_support_params(
            key_support, M_parts, N_per_part, grid_shape, size_factor,
            use_gates=use_gates,
        )
        support_static = {"support_type": "multi", "use_gates": use_gates}
    elif support_type == "freeform":
        init_logit = support_kwargs.get("init_logit", None)
        noise_scale = support_kwargs.get("noise_scale", 0.0)
        support_params = init_freeform_support_params(
            key_support, grid_shape, init_logit=init_logit, noise_scale=noise_scale
        )
        support_static = {"support_type": "freeform"}
    else:
        raise ValueError(
            f"Unknown support_type: {support_type!r}, choose 'single', 'multi' or 'freeform'."
        )

    phase_params, phase_static = init_phase_params(
        key_phase, grid_shape, phase_type=phase_type, **(phase_kwargs or {})
    )
    model_static = dict(phase_static)
    model_static.update(support_static)

    params = {"support": support_params, "phase": phase_params}
    return params, model_static


def init_params_only(key, grid_shape, N=64, size_factor=4.0, phase_type="grid",
                      phase_kwargs=None, support_type="single", support_kwargs=None):
    """Like `init_model`, but drops the (non-batchable, string-containing)
    static metadata -- for use under `jax.vmap` when generating a population
    of restarts. `model_static` is identical for every restart by
    construction (it never depends on the random key), so callers should
    fetch it once via `init_model` outside the vmap.
    """
    params, _model_static = init_model(
        key, grid_shape, N=N, size_factor=size_factor, phase_type=phase_type,
        phase_kwargs=phase_kwargs, support_type=support_type,
        support_kwargs=support_kwargs,
    )
    return params


def _amplitude_from(support, Iobs, eps=1e-12, stop_grad=False):
    N = Iobs.size
    sum_I = jnp.sum(Iobs)
    sum_S = jnp.sum(support ** 2)
    amplitude = jnp.sqrt(sum_I / (N * sum_S + eps))
    if stop_grad:
        amplitude = jax.lax.stop_gradient(amplitude)
    return amplitude


def forward(params, coords, Iobs, eps, model_static, stop_amplitude_grad=False):
    """Single-instance forward pass.

    Returns (support, amplitude, phase_or_phasor) exactly like the original
    `PhaseRetrievalModel.call`.
    """
    support_type = model_static.get("support_type", "single")
    if support_type == "multi":
        support = compute_multi_support(
            params["support"], coords, eps, use_gates=model_static.get("use_gates", False)
        )
    elif support_type == "freeform":
        # `eps` doubles as `T` here -- same "boundary softness, anneal it
        # down" role it plays for compute_support/compute_multi_support,
        # just without any half-space geometry behind it. `coords` is
        # unused: every voxel already owns its own parameter.
        support = compute_freeform_support(params["support"], T=eps)
    else:
        support = compute_support(params["support"], coords, eps)

    amplitude = _amplitude_from(support, Iobs, stop_grad=stop_amplitude_grad)

    # Bug fixed here, found by re-checking against the original phase.py:
    # the original dispatches on `hasattr(self.phaser, "compute_phasor")`,
    # and BOTH GridPhasor and DisplacementPhasor define that method (only
    # GridPhase doesn't) -- so the original ALWAYS uses the (cos, sin)
    # tuple form for 'phasor' AND 'displacement', never the raw scalar
    # phase for either. An earlier version of this file only did this for
    # 'phasor', silently falling back to the scalar form for
    # 'displacement'. The two forms give identical values in fourier_loss
    # (cos/sin either way), but they give a DIFFERENT phase-TV regularizer
    # in losses.tv_loss_phase (raw (c,s)-difference smoothness vs.
    # atan2-based wrapped-phase smoothness) -- so this only mattered when
    # beta > 0, but it is a real behavioral mismatch worth fixing.
    if model_static["phase_type"] in ("phasor", "displacement"):
        phase = compute_phasor(params["phase"], model_static)
    else:
        phase = compute_phase(params["phase"], model_static)

    return support, amplitude, phase


def loss_fn(params, static):
    """Scalar loss for one instance; this is what gets passed to the
    optimizer (see optimize.py). `static` bundles everything the optimizer
    should treat as constant.

    static = {
        "coords": (D, H, W, 3),
        "Iobs": (Do, Ho, Wo),
        "eps": float,               # half-space softness, or freeform T
        "alpha": float,             # support-size weight
        "beta": float,              # phase-TV weight
        "metric": "mae" | "poisson",
        "phase_static": {...},      # from init_model, incl. support_type/use_gates
        "gamma": float,             # optional: freeform perimeter weight (default 0)
        "delta": float,             # optional: freeform double-well weight (default 0)
        "zeta": float,              # optional: freeform anchor weight (default 0)
        "S_ref": (D, H, W) | None,  # optional: freeform anchor target, required if zeta != 0
    }
    The last four keys only matter for support_type == "freeform" (see
    support_freeform.py); they default to 0/None so existing "single"/
    "multi" call sites that never set them are unaffected.
    """
    support, amplitude, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"], static["phase_static"], stop_amplitude_grad=static.get("stop_amplitude_grad", False),
    )
    return total_loss(
        support, amplitude, phase, static["Iobs"],
        alpha=static["alpha"], beta=static["beta"], metric=static["metric"],
        gamma=static.get("gamma", 0.0), delta=static.get("delta", 0.0),
        zeta=static.get("zeta", 0.0), S_ref=static.get("S_ref", None),
    )
