# =============================================================================
# MODEL
# =============================================================================
# Functional, single-instance model: `params` is a plain dict pytree and
# everything that needs no gradient (Iobs, coords, eps, loss weights,
# phase_type, ...) lives in a separate `static` dict. `optimize.py` vmaps
# these functions over a leading population axis of `params`.
import jax
import jax.numpy as jnp

from .support import compute_support, init_support_params, make_coords
from .support_freeform import init_freeform_support_params, compute_freeform_support
from .phase import compute_phase, compute_phasor, init_phase_params
from .losses import total_loss


def make_coords_for(Iobs_shape):
    """grid_shape = Iobs // 2, plus the fixed coordinate grid for it."""
    D, H, W = Iobs_shape
    grid_shape = (D // 2, H // 2, W // 2)
    return grid_shape, make_coords(grid_shape)


def init_model(key, grid_shape, N=64, size_factor=4.0, phase_type="grid",
               phase_kwargs=None, support_type="single", support_kwargs=None):
    """Initialize a single instance's trainable parameters.

    support_type : "single" (default) | "freeform"
        "single": one convex object, `N` half-spaces (params {"v", "d"}).
        "freeform": one free logit per voxel (see support_freeform.py).
            Intended as a stage-2 release from a converged "single" solution.
            `support_kwargs` may contain "init_logit" ((D,H,W) array, typically
            `invert_support_to_logit(S_stage1)`) and "noise_scale" (default 0.0).
        (The "multi" support was removed.)

    Returns
    -------
    params : {"support": ..., "phase": ...}
    model_static : non-trainable metadata ("phase_type", ..., "support_type").
    """
    key_support, key_phase = jax.random.split(key)
    support_kwargs = support_kwargs or {}

    if support_type == "single":
        support_params = init_support_params(
            key_support, N, grid_shape, size_factor,
            d_jitter=support_kwargs.get("d_jitter", 0.05),
        )
        support_static = {"support_type": "single"}
    elif support_type == "freeform":
        support_params = init_freeform_support_params(
            key_support, grid_shape,
            init_logit=support_kwargs.get("init_logit", None),
            noise_scale=support_kwargs.get("noise_scale", 0.0),
        )
        support_static = {"support_type": "freeform"}
    else:
        raise ValueError(
            f"Unknown support_type: {support_type!r}, choose 'single' or 'freeform'."
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
    """Like `init_model` but drops the (string-containing) static metadata, for
    use under `jax.vmap`. Fetch the metadata once via `init_model`."""
    params, _ = init_model(
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
    """Single-instance forward pass -> (support, amplitude, phase_or_phasor)."""
    support_type = model_static.get("support_type", "single")
    if support_type == "freeform":
        # `eps` doubles as `T`: boundary softness, anneal it down. `coords` is
        # unused: every voxel owns its own parameter.
        support = compute_freeform_support(params["support"], T=eps)
    else:
        support = compute_support(params["support"], coords, eps)

    amplitude = _amplitude_from(support, Iobs, stop_grad=stop_amplitude_grad)

    # 'phasor' AND 'displacement' always use the (cos, sin) form, as in the
    # original implementation (this matters for the phase-TV regularizer).
    if model_static["phase_type"] in ("phasor", "displacement"):
        phase = compute_phasor(params["phase"], model_static)
    else:
        phase = compute_phase(params["phase"], model_static)

    return support, amplitude, phase


def loss_fn(params, static):
    """Scalar loss for one instance.

    static = {
        "coords": (D, H, W, 3), "Iobs": (Do, Ho, Wo),
        "eps": float,            # half-space softness, or freeform T
        "alpha": float,          # support-size weight
        "beta": float,           # phase-TV weight
        "metric": "mae" | "poisson",
        "phase_static": {...},   # from init_model
        optional: "stop_amplitude_grad",
                  "gamma", "delta", "zeta", "S_ref"   (freeform only)
    }
    """
    support, amplitude, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"],
        static["phase_static"],
        stop_amplitude_grad=static.get("stop_amplitude_grad", False),
    )
    return total_loss(
        support, amplitude, phase, static["Iobs"],
        alpha=static["alpha"], beta=static["beta"], metric=static["metric"],
        gamma=static.get("gamma", 0.0), delta=static.get("delta", 0.0),
        zeta=static.get("zeta", 0.0), S_ref=static.get("S_ref", None),
    )