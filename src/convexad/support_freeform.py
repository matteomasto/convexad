# =============================================================================
# FREE-FORM (VOXEL-GRID) SUPPORT
# =============================================================================
# Stage-2 support parameterization for objects that the convex (support.py)
# and multi-convex (multi_support.py) parameterizations cannot represent
# exactly: twin-boundary facets, concave notches, multiple particles with
# irregular shapes. Instead of N half-space planes (or M unions of them),
# every voxel gets its own free logit:
#
#     S(x) = sigmoid(logit(x) / T)
#
# `T` plays the same role `eps` plays for `compute_support`: it sets the
# softness of the S -> {0,1} transition and is meant to be annealed down
# over the course of optimization, not fixed as in the current
# single/multi-convex API.
#
# Because this parameterization has D*H*W free parameters (millions, at
# realistic BCDI grid sizes) instead of O(N) or O(M*N), it is NOT meant to
# be run from a cold random start -- nothing here constrains the result to
# look like a single compact object, so an unconstrained solve from noise
# will happily fit speckle in Iobs. It is meant to be warm-started from a
# converged convex or multi-convex solution via `invert_support_to_logit`,
# kept close to that solution early on via `anchor_penalty`, and shaped by
# `tv_support` + `double_well_support` throughout. All of this exists to be
# driven by a two-stage reconstruction loop (see optimize.py /
# reconstruct_two_stage): stage 1 (support.py or multi_support.py) finds
# the rough global shape and phase field cheaply; stage 2 (this module)
# releases the support to free-form and refines it.
#
# All shapes below are for a SINGLE reconstruction instance (no leading
# batch axis), matching support.py / multi_support.py -- a population of
# restarts is handled by vmapping over this module's functions from
# optimize.py.
#
# No custom_vjp here, unlike support.py: nothing below materializes a
# (D,H,W,N)-shaped intermediate (every voxel already owns exactly one
# parameter), so ordinary JAX autodiff on these elementwise/differencing
# ops is already O(D*H*W) -- there is nothing a hand-written adjoint would
# save.
# =============================================================================
import jax
import jax.numpy as jnp


def init_freeform_support_params(key, grid_shape, init_logit=None, noise_scale=0.0):
    """Init a single instance's free-form support parameters.

    Parameters
    ----------
    grid_shape : (D, H, W).
    init_logit : optional (D, H, W) array to warm-start from -- typically
        `invert_support_to_logit(S_stage1)`, i.e. the converged
        single/multi-convex support from stage 1. If None, starts from an
        all-neutral field (logit = 0, i.e. S = 0.5 everywhere at T=1) plus
        optional noise. A cold (all-neutral or random) start has no
        incentive to form a compact object at all -- see module
        docstring -- so leaving `init_logit=None` is only appropriate for
        small debugging grids or unit tests, not real reconstructions.
    noise_scale : stddev of Gaussian noise added on top of `init_logit` (or
        the zero field). Used to decorrelate a population of restarts that
        all share the same stage-1 warm start.

    Returns
    -------
    params : {"logit": (D, H, W) array}.
    """
    grid_shape = tuple(grid_shape)
    if init_logit is None:
        logit0 = jnp.zeros(grid_shape, dtype=jnp.float32)
    else:
        logit0 = jnp.asarray(init_logit, dtype=jnp.float32)
        if logit0.shape != grid_shape:
            raise ValueError(
                f"init_logit shape {logit0.shape} does not match grid_shape {grid_shape}"
            )
    if noise_scale > 0.0:
        logit0 = logit0 + noise_scale * jax.random.normal(key, grid_shape)
    return {"logit": logit0}


def invert_support_to_logit(S, eps=1e-4):
    """Invert a converged soft support S (from `compute_support` or
    `compute_multi_support`) into a free-form logit field that reproduces
    it exactly at T=1:

        sigmoid(logit) = S   =>   logit = log(S / (1 - S))

    `eps` clips S away from {0, 1} first -- the logit is +-inf there, which
    would poison gradients at every voxel that started fully saturated
    (exactly the interior/exterior bulk of a typical support, i.e. most of
    the volume). Clipping trades a small, deliberate loss of confidence in
    the warm start's most saturated voxels for a finite, well-conditioned
    gradient everywhere.
    """
    S = jnp.clip(jnp.asarray(S, dtype=jnp.float32), eps, 1.0 - eps)
    return jnp.log(S / (1.0 - S))


def compute_freeform_support(params, T=1.0):
    """S(x) = sigmoid(logit(x) / T). No coords/geometry input needed --
    unlike the half-space supports, every voxel is already its own
    parameter.

    T anneals the same way `eps` does for `compute_support`: large T gives
    a smooth/soft boundary (a favorable, low-curvature early-training
    landscape where gradients can move mass around freely); T -> small
    sharpens it toward a near-binary mask.
    """
    return jax.nn.sigmoid(params["logit"] / T)


# =============================================================================
# REGULARIZERS
# =============================================================================
# Meant to be added into losses.total_loss alongside the existing
# small_support / tv_loss_phase terms whenever support_type == "freeform".
# small_support(support) (losses.py) still applies unchanged -- it penalizes
# total mass, which is orthogonal to everything below.

def tv_support(S):
    """Perimeter-like regularizer: mean squared finite difference of S
    along each axis.

    Penalizes ANY boundary -- isolated speckle voxels are disproportionately
    expensive per unit volume, since a lone voxel has the most surface area
    per unit mass -- without penalizing concavity specifically: a
    twin-boundary facet or a notch between two particles costs exactly as
    much perimeter per unit area as a convex bulge does. This is the
    free-form complement to `small_support` (losses.py): that penalizes how
    much mass the support has, this penalizes how fragmented/rough its
    boundary is.
    """
    dz = jnp.diff(S, axis=0)
    dy = jnp.diff(S, axis=1)
    dx = jnp.diff(S, axis=2)
    return jnp.mean(dz ** 2) + jnp.mean(dy ** 2) + jnp.mean(dx ** 2)


def double_well_support(S):
    """Modica-Mortola-style double-well term: mean(S^2 * (1-S)^2).

    Minimized at S in {0, 1}, maximized at S = 0.5. Paired with
    `tv_support`, this is a discrete phase-field (Allen-Cahn /
    Ginzburg-Landau) energy whose T -> 0 limit is perimeter minimization
    subject to the data term -- i.e. this pair turns "penalize gradients"
    into "penalize gradients of an otherwise genuinely-binary mask" instead
    of just rewarding a globally blurred S. Without this term, `tv_support`
    alone can be trivially minimized by flattening S toward a uniform
    mid-value, which is not what we want.
    """
    return jnp.mean((S ** 2) * (1.0 - S) ** 2)


def anchor_penalty(S, S_ref):
    """Pulls S toward a fixed reference support, mean((S - S_ref)^2).

    `S_ref` is typically the stage-1 (single/multi-convex) solution, and
    should be `jax.lax.stop_gradient`'d by the caller if it was produced
    inside the same traced computation -- it is a target, not something to
    differentiate through.

    Meant to be scaled by a weight that the driver decays over stage 2:
    large early (right after switching parameterizations, when per-voxel
    gradients are least trustworthy and the free-form field could
    otherwise run off in an unphysical direction) and weak or zero later
    (once the free-form boundary should be trusted to add genuine
    non-convex detail -- a twin facet, a second particle, a concave notch
    -- that the stage-1 solution structurally could not represent).
    """
    return jnp.mean((S - S_ref) ** 2)
