"""Chunked, interruptible and resumable reconstruct()."""
import os
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from convexad import optimize
from convexad.model import loss_fn
from convexad.optimize import reconstruct

IOBS = np.random.default_rng(0).random((12, 10, 8)).astype(np.float32) + 0.2
KEY = jax.random.PRNGKey(0)
KW = dict(
    n_restarts=3, N=4, eps=0.6, alpha=0.01, beta=0.01, metric="poisson",
    phase_type="displacement", phase_kwargs={"hkl": [1, 1, 1]},
    tol=1e-12, verbose=False,
)


def run(**overrides):
    return reconstruct(KEY, IOBS, **{**KW, **overrides})


def assert_same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)


def max_change(a, b):
    return max(
        float(jnp.max(jnp.abs(x - y)))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_chunking_does_not_change_the_trajectory():
    whole = run(max_steps=30, chunk=1000)
    chunked = run(max_steps=30, chunk=7)
    assert whole.all_steps.tolist() == chunked.all_steps.tolist() == [30, 30, 30]
    assert_same(whole.all_params, chunked.all_params)
    assert_same(whole.all_losses, chunked.all_losses)


def test_resume_equals_an_uninterrupted_run():
    """Only true if params, optimizer state and step count are all carried over."""
    whole = run(max_steps=30)
    first = run(max_steps=12)
    assert first.all_steps.tolist() == [12, 12, 12]
    second = run(max_steps=30, resume=first)
    assert second.all_steps.tolist() == [30, 30, 30]
    assert_same(whole.all_params, second.all_params)
    assert_same(whole.all_losses, second.all_losses)


def test_edited_hyperparameters_are_used_after_resume():
    first = run(max_steps=12)
    edited = dict(eps=0.9, alpha=0.05, beta=0.02, learning_rate=0.01, max_steps=24)
    second = run(resume=first, **edited)
    unedited = run(resume=first, max_steps=24)

    assert second.all_steps.tolist() == [24, 24, 24]
    assert second.eps == 0.9
    assert max_change(second.all_params, unedited.all_params) > 1e-4

    # The reported loss is the loss under the NEW eps, alpha and beta.
    static = {
        "coords": second.coords, "Iobs": jnp.asarray(IOBS), "eps": 0.9, "alpha": 0.05,
        "beta": 0.02, "metric": "poisson", "phase_static": second.model_static,
    }
    np.testing.assert_allclose(
        float(second.best_loss), float(loss_fn(second.best_params, static)), rtol=1e-5
    )


def test_learning_rate_edit_takes_effect():
    first = run(max_steps=12)
    frozen = run(resume=first, max_steps=24, learning_rate=1e-9)
    moving = run(resume=first, max_steps=24, learning_rate=0.05)
    assert max_change(first.all_params, frozen.all_params) < 1e-6
    assert max_change(first.all_params, moving.all_params) > 1e-3


def test_editing_hyperparameters_does_not_recompile():
    cache_size = getattr(optimize._advance_population, "_cache_size", None)
    if cache_size is None:
        pytest.skip("this JAX version does not expose the jit cache size")
    first = run(max_steps=12)
    n = cache_size()
    run(resume=first, max_steps=40, eps=0.9, alpha=0.05, beta=0.0, learning_rate=0.01, tol=1e-9)
    assert cache_size() == n


def _interrupt_while_waiting_for_chunk(monkeypatch, n, chunk_finished, real_signal=False):
    """Interrupt while the n-th chunk is running, as the kernel's stop button does."""
    real_wait, calls = optimize._wait, []

    def fake(x):
        calls.append(1)
        if len(calls) == n:
            monkeypatch.setattr(optimize, "_is_ready", lambda _: chunk_finished)
            if real_signal:
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(5)          # the signal is delivered here
            raise KeyboardInterrupt
        return real_wait(x)

    monkeypatch.setattr(optimize, "_wait", fake)


@pytest.mark.parametrize("real_signal", [False, True], ids=["raised", "sigint"])
def test_interrupt_returns_the_last_finished_chunk_to_inspect_and_resume(monkeypatch, real_signal):
    whole = run(max_steps=30)
    with monkeypatch.context() as m:
        _interrupt_while_waiting_for_chunk(m, 3, chunk_finished=False, real_signal=real_signal)
        partial = run(max_steps=30, chunk=5)

    assert partial.all_steps.tolist() == [10, 10, 10]      # two chunks done, the third dropped
    support, amplitude, phase = partial.evaluate(IOBS)      # inspectable
    assert np.all(np.isfinite(np.asarray(support)))
    assert np.all(np.isfinite(np.asarray(partial.all_losses)))

    resumed = run(max_steps=30, resume=partial)
    assert resumed.all_steps.tolist() == [30, 30, 30]
    assert_same(whole.all_params, resumed.all_params)


def test_interrupt_keeps_the_running_chunk_if_it_had_just_finished(monkeypatch):
    whole = run(max_steps=30)
    with monkeypatch.context() as m:
        _interrupt_while_waiting_for_chunk(m, 3, chunk_finished=True)
        partial = run(max_steps=30, chunk=5)
    assert partial.all_steps.tolist() == [15, 15, 15]
    assert_same(whole.all_params, run(max_steps=30, resume=partial).all_params)


def test_interrupt_before_any_chunk_is_dispatched_returns_the_initial_state(monkeypatch):
    """E.g. the kernel is interrupted while the first chunk is still being compiled."""
    whole = run(max_steps=30)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(optimize, "_advance_population", interrupted)
        initial = run(max_steps=30)

    assert initial.all_steps.tolist() == [0, 0, 0]
    assert np.all(np.isnan(np.asarray(initial.all_losses)))         # nothing evaluated yet
    initial.evaluate(IOBS)                                          # still inspectable

    resumed = run(max_steps=30, resume=initial)
    assert resumed.all_steps.tolist() == [30, 30, 30]
    assert_same(whole.all_params, resumed.all_params)


def test_resume_needs_an_optimizer_state():
    with pytest.raises(ValueError, match="optimizer state"):
        run(max_steps=5, resume=run(max_steps=5)._replace(opt_state=None))
