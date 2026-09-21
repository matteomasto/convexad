# ConvexAD

<img width="1240" height="820" alt="image" src="https://github.com/user-attachments/assets/df4366b5-9a40-466b-a82e-a13eff3cfae0" />

---
Geometrically regularized automatic differentiation framework for BCDI phase retrieval.
---

ConvexAD is a gradient based phase retrieval tool for Bragg coherent diffraction imaging (BCDI), built on JAX with native GPU acceleration. Instead of alternating projections, it encodes prior knowledge in a differentiable model of the object and fits the measured intensity by gradient descent.

* **Support.** A convex polytope, the soft intersection of `N` half spaces. Non convex objects are modeled as a union of convex parts.
* **Phase.** Free per voxel, a phasor, or a displacement field along the scattering vector.
* **Amplitude.** Constant inside the support and set analytically from Parseval's theorem, not trained.
* **Restarts.** Many random restarts run in parallel and the one with the lowest loss is kept.
* **Optimizers.** AMSGrad, AdaBelief or Lion, with a decaying learning rate and optional gradient clipping.
* **Pause and resume.** Interrupt a run, inspect the result, change `eps`, `alpha`, `beta` or the learning rate, and continue.

## Installation

ConvexAD needs Python 3.10 or newer, JAX, optax 0.2.5 or newer and NumPy. It is developed with JAX 0.6.2, optax 0.2.6 and NumPy 2.2.6 on an NVIDIA A40 GPU.

```bash
git clone https://github.com/matteomasto/convexad.git
cd convexad
pip install -e ".[examples]"
```

The `examples` extra installs `matplotlib` and `scikit-image` for the plotting helpers, and `hdf5plugin` for compressed HDF5 files. This gives JAX on the CPU, which is enough for small problems. For a GPU, install JAX with CUDA support first, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The refinement step of the demo notebook uses [PyNX](https://gitlab.esrf.fr/favre/PyNX) and `cdiutils`, which are installed separately. `install_env.sh` builds the full environment used at ESRF (a virtual environment with JAX, PyNX, `cdiutils` and ConvexAD). It is site specific, so edit the paths at the top before using it.

## Quick start

`Iobs` is the measured 3D intensity (not the amplitude) as a float32 array with the Bragg peak at the center. The object is reconstructed on a grid of half the data size along each axis, which corresponds to an oversampling of 2. Use `grid_shape` to change it.

```python
import jax
import numpy as np
from convexad import reconstruct

Iobs = np.load("data.npz")["I"].astype(np.float32)
key = jax.random.PRNGKey(0)

result = reconstruct(
    key, Iobs,
    n_restarts=32, N=64,
    eps=0.8, alpha=0.0, beta=0.05,
    metric="mae",
    phase_type="displacement", phase_kwargs={"hkl": [2, 2, 2]},
    max_steps=1000, clip_norm=1.0,
)

support, amplitude, phase = result.evaluate(Iobs)      # best restart
modulus = amplitude * support
obj = modulus * (phase[0] + 1j * phase[1])              # displacement and phasor phases are (cos, sin) pairs
# with phase_type="grid": obj = modulus * np.exp(1j * phase)
```

`result` holds `best_loss`, `best_params` and, for every restart, `all_losses` and `all_steps`. To look at the reconstruction:

```python
from convexad.viz import plot_2D_slices_middle_only_module, plot_2D_slices_middle_only_phase

plot_2D_slices_middle_only_module(obj)
plot_2D_slices_middle_only_phase(obj, unwrap=False)
```

`notebooks/demo.ipynb` shows the complete workflow, including the refinement with PyNX. It loads the intensity from an npz file with the key `I`, so bring your own data.

## Pause, inspect and resume

`reconstruct` advances `chunk` steps at a time (default 100) and prints its progress. Interrupt the kernel, or press Ctrl+C, at any time. It then returns immediately, instead of raising, with the result as of the last progress line. Inspect it, then continue from the same parameters and optimizer state with `resume`. You can edit `eps`, `alpha`, `beta`, `learning_rate`, `tol` and `max_steps` between runs, and none of these triggers a recompilation.

```python
settings = dict(
    n_restarts=32, N=64, metric="mae",
    phase_type="displacement", phase_kwargs={"hkl": [2, 2, 2]}, clip_norm=1.0,
)
result = reconstruct(key, Iobs, eps=0.8, alpha=0.0, beta=0.05, max_steps=1000, **settings)

# Interrupt at any time, inspect `result`, then continue with new values:
result = reconstruct(
    key, Iobs, eps=0.6, alpha=0.0, beta=0.02, learning_rate=0.02, max_steps=2000,
    resume=result, **settings,
)
```

* `max_steps` is the total number of steps, not an increment.
* `learning_rate` is the base of the decay schedule, which continues from the current step.
* All other arguments must stay as in the first call. When resuming, `key`, `n_restarts` and the model arguments are ignored.
* If you interrupt before the first chunk has finished, the result is the initial state, with NaN losses. It can still be resumed.

## Main options

| Argument | Meaning |
|---|---|
| `n_restarts` | Number of independent random restarts run in parallel. |
| `N` | Number of half spaces of the convex support. |
| `support_type` | `"single"` for one convex object, or `"multi"` for a union of convex parts, with `support_kwargs={"M_parts": 2, "N_per_part": 16}`. |
| `phase_type` | `"grid"`, `"phasor"` or `"displacement"`. The last one needs `phase_kwargs={"hkl": [h, k, l]}`, optionally with `lattice_matrix`. |
| `metric` | `"mae"`, `"mse"` or `"poisson"`. |
| `eps` | Width of the support boundary, in voxels of the object grid. |
| `alpha` | Weight of the support size penalty. |
| `beta` | Weight of the total variation penalty on the phase. |
| `variant` | `"amsgrad"`, `"adabelief"` or `"lion"`. |
| `learning_rate`, `decay_steps`, `decay_rate` | The learning rate is `learning_rate * decay_rate ** floor(step / decay_steps)`. |
| `clip_norm` | Clips the global gradient norm to this value. |
| `max_steps`, `tol` | Maximum number of steps. A restart stops when its gradient norm falls below `tol`. |
| `grid_shape` | Object grid. Default: half the shape of `Iobs`. |
| `resume`, `chunk` | Continue a previous result, and the number of steps between two checks. |

The defaults of `alpha` and `beta` are 0.8 and 0.1, so set them explicitly. See `help(reconstruct)` for the full list.

## How it works

* **Support.** `S(x) = prod_i sigmoid((d_i - n_i . x) / eps)` is the soft intersection of `N` half spaces with offsets `d_i` and unit normals `n_i`. Each normal is parametrized by a stereographic projection, so it has two free parameters and no redundant direction.
* **Memory.** Evaluating all planes at once would create a `(D, H, W, N)` tensor. At 125x225x225 voxels and `N = 256` this is about 6 GiB in float32 for one tensor. The support has a hand written adjoint that scans over the planes and recomputes each sigmoid in the backward pass, so the extra memory is `O(D*H*W)` and does not depend on `N`.
* **Amplitude.** Parseval's theorem fixes the amplitude from the total measured intensity and the support, so it is not a parameter. The Fourier fidelity term needs no custom derivative, because the FFT is linear and JAX already provides its adjoint.
* **Restarts.** The physics and model code work on a single reconstruction. The population of restarts is added with `jax.vmap` in `optimize.py`, including through the custom adjoint.

## Package layout

```
src/convexad/
    support.py               convex support and its adjoint
    multi_support.py         non convex support as a union of convex parts
    support_freeform.py      per voxel support (experimental)
    phase.py                 grid, phasor and displacement phases
    losses.py                mae, mse, Poisson, phase total variation, support size
    model.py                 single instance init, forward and loss
    optimize.py              restarts with vmap, optimizers, pause and resume
    viz.py                   plotting helpers
    pynx_reconstruction.py   helpers to refine a ConvexAD object with PyNX
    utilities.py             centering, cropping and GPU memory helpers
notebooks/
    demo.ipynb               complete workflow
```

## Related

* PyNX, used for the refinement step: V. Favre-Nicolin et al., J. Appl. Cryst. 53, 1404 (2020).
* A TensorFlow implementation is kept at [convexad_tf](https://github.com/matteomasto/convexad_tf).

## License

MIT, see [LICENSE](LICENSE).
