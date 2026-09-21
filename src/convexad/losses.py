# =============================================================================
# LOSSES
# =============================================================================
# Single-instance (no leading batch axis). The FFT-based fidelity term is
# left to JAX's built-in autodiff: jnp.fft is linear, so its VJP is just an
# (adjoint) FFT with no extra activation storage -- there is nothing a
# hand-written adjoint would improve here.
import jax
import jax.numpy as jnp

from .support_freeform import tv_support, double_well_support, anchor_penalty


def mae(Iobs, Icalc):
    """Normalized mean absolute error."""
    return jnp.sum(jnp.abs(Iobs - Icalc)) / jnp.sum(Iobs)

def mse(Iobs, Icalc):
    """Normalized mean squared error."""
    return jnp.sum((jnp.sqrt(Iobs) - jnp.sqrt(Icalc))**2) / jnp.sum(jnp.sqrt(Iobs))
    
def poisson_kl(Iobs, Icalc, eps=1e-12):
    """Poisson KL divergence (generalized, a.k.a. I-divergence), averaged
    per voxel.

    Uses jax.scipy.special.kl_div(Iobs, Icalc_safe) directly -- this IS
    the formula (Icalc - Iobs + Iobs*log(Iobs/Icalc)) for Iobs>0,Icalc>0;
    reduces correctly to Icalc_safe at Iobs=0, Icalc_safe>=0. Unlike a
    hand-rolled xlogy(Iobs, ratio) composition, kl_div's implementation
    masks BOTH operands together via lax.select before any log/div runs,
    built from ordinary primitives (mul, log, div, select) rather than
    going through xlogy's own custom_jvp rule -- avoids that class of
    fragility entirely rather than working around it.

    eps floors Icalc at a physically meaningful scale (default 1.0, ~one
    photon count) rather than a numerical-safety epsilon: sub-single-
    photon intensities aren't statistically meaningful for counting data,
    and d(kl)/dIcalc ~ Iobs/Icalc diverges as Icalc->0 with Iobs>0 --
    the same curvature blowup already floored in _newton_step_size's
    Hessian terms; this is the identical divergence in the loss's own
    gradient, not covered by kl_div's own (Iobs=0) zero-handling.
    """
    Icalc_safe = jnp.clip(Icalc, eps, None)
    kl = jax.scipy.special.kl_div(Iobs, Icalc_safe)
    N = Iobs.size
    return jnp.sum(kl) / N


def _center_pad(obj, target_shape):
    pads = []
    for src, dst in zip(obj.shape, target_shape):
        delta = dst - src
        p0 = delta // 2
        p1 = delta - p0
        pads.append((p0, p1))
    return jnp.pad(obj, pads)


def compute_Icalc(support, amplitude, phase, Iobs):
    """Forward model only (no metric reduction) -- factored out of
    fourier_loss so the GN/LM residual can reuse it without duplicating
    the FFT logic."""
    modulus = support * amplitude
    if isinstance(phase, tuple):
        c, s = phase
        obj = jax.lax.complex(modulus * c, modulus * s)
    else:
        obj = jax.lax.complex(modulus * jnp.cos(phase), modulus * jnp.sin(phase))
    obj_p = _center_pad(obj, Iobs.shape)
    Icalc = jnp.abs(jnp.fft.ifftshift(jnp.fft.fftn(jnp.fft.fftshift(obj_p)))) ** 2
    return Icalc.astype(jnp.float32)


def fourier_loss(support, amplitude, phase, Iobs, metric="mae"):
    Iobs = Iobs.astype(jnp.float32)
    Icalc = compute_Icalc(support, amplitude, phase, Iobs)
    if metric == "mae":
        return mae(Iobs, Icalc)
    if metric == "mse":
        return mse(Iobs, Icalc)
    if metric == "poisson":
        return poisson_kl(Iobs, Icalc)
    raise ValueError(f"Unknown metric: {metric!r}, choose 'mae', 'mse' or 'poisson'.")


def tv_loss_phase(phase, eps=1e-9):
    """Total variation of the phase field (scalar phase or (c, s) phasor)."""

    def diff(x, axis):
        return jnp.diff(x, axis=axis)

    if isinstance(phase, tuple):
        c, s = phase
        dcx, dcy, dcz = diff(c, 0), diff(c, 1), diff(c, 2)
        dsx, dsy, dsz = diff(s, 0), diff(s, 1), diff(s, 2)

        dcx2 = jnp.square(dcx[:, :-1, :-1])
        dcy2 = jnp.square(dcy[:-1, :, :-1])
        dcz2 = jnp.square(dcz[:-1, :-1, :])

        dsx2 = jnp.square(dsx[:, :-1, :-1])
        dsy2 = jnp.square(dsy[:-1, :, :-1])
        dsz2 = jnp.square(dsz[:-1, :-1, :])

        grad_sq = (dcx2 + dcy2 + dcz2) + (dsx2 + dsy2 + dsz2)
        return jnp.mean(grad_sq + eps)

    phi = phase

    def wrapped_diff(p, axis):
        d = jnp.diff(p, axis=axis)
        return jnp.arctan2(jnp.sin(d), jnp.cos(d))

    dx = wrapped_diff(phi, 0)
    dy = wrapped_diff(phi, 1)
    dz = wrapped_diff(phi, 2)

    dx2 = jnp.square(dx[:, :-1, :-1])
    dy2 = jnp.square(dy[:-1, :, :-1])
    dz2 = jnp.square(dz[:-1, :-1, :])

    grad_sq = dx2 + dy2 + dz2
    return jnp.mean(grad_sq + eps)


def small_support(support):
    """Penalty on the size (mass) of the support."""
    return jnp.sum(support)


def total_loss(
    support, amplitude, phase, Iobs, alpha=0.8, beta=0.1, metric="mae",
    gamma=0.0, delta=0.0, zeta=0.0, S_ref=None,
):
    """Fourier fidelity + support-size penalty + phase-TV penalty, plus
    optional free-form support regularizers (support_freeform.py).

    Parameters
    ----------
    alpha : weight for the support-size penalty.
    beta  : weight for the phase TV penalty.
    gamma : weight for `tv_support` (perimeter-like boundary regularizer).
        0 by default -- a no-op for "single"/"multi" call sites that never
        pass it.
    delta : weight for `double_well_support` (pushes S toward {0, 1}).
        0 by default, same reasoning as `gamma`.
    zeta  : weight for `anchor_penalty` toward `S_ref`. 0 by default.
    S_ref : optional (D, H, W) reference support to anchor toward (see
        `support_freeform.anchor_penalty`); required if `zeta != 0`. Not
        differentiated through -- stop_gradient'd here defensively even
        though callers should already be passing a constant.

    `gamma`/`delta`/`zeta` are defined generically here (nothing below
    checks what produced `support`), so they COULD also lightly regularize
    a "single"/"multi" convex support if ever useful -- they aren't tied to
    support_type == "freeform" by construction, only by convention (see
    model.loss_fn).
    """
    fourier = fourier_loss(support, amplitude, phase, Iobs, metric=metric)
    small = alpha * small_support(support)
    smooth_phase = beta * tv_loss_phase(phase)
    total = fourier + small + smooth_phase
    if gamma != 0.0:
        total = total + gamma * tv_support(support)
    if delta != 0.0:
        total = total + delta * double_well_support(support)
    if zeta != 0.0:
        total = total + zeta * anchor_penalty(support, jax.lax.stop_gradient(S_ref))
    return total
