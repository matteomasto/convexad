# =============================================================================
# VISUALIZATION UTILITIES (JAX port of convex_ad.viz)
# =============================================================================
# All plotting here works on plain numpy arrays. JAX arrays (and any
# array-like with __array__) are converted up front with np.asarray, so
# these functions accept jax.numpy arrays, numpy arrays, or the outputs of
# ReconstructionResult.evaluate(...) / result.best_params interchangeably --
# no jax import is required in this module.
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _to_numpy(array):
    """Convert a jax.numpy / numpy / array-like to a plain numpy array."""
    if array is None:
        return None
    return np.asarray(array)


def crop_tensor_half_size(tensor: np.ndarray) -> np.ndarray:
    """
    Crop spatial dimensions by factor 2 (centered).

    Assumes shape: (B, D, H, W)

    Returns
    -------
    np.ndarray
        Cropped tensor of shape (B, D/2, H/2, W/2)
    """
    tensor = _to_numpy(tensor)
    _, D, H, W = tensor.shape
    return tensor[
        :,
        D // 4:3 * D // 4,
        H // 4:3 * H // 4,
        W // 4:3 * W // 4,
    ]


def plot_module_histogram(obj,
                          crop=True,
                          bins=None, 
                          fig=None, ax=None):
    module, phase = get_cropped_module_phase(obj, crop=crop)
    module[module<.01*np.max(module)] = np.nan
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(6,4))
        
    if bins is None:
        if obj.ndim == 2:
            bins = 30
        if obj.ndim == 3:
            bins = 50
    n, bins, patches = ax.hist(module.flatten(), bins=bins)

    ax.set_xlabel('object module value', fontsize=15)
    ax.set_ylabel('number of pixels', fontsize=15)
    return

def plot_2D_slices_middle(obj,
                          crop=False,
                          support=None, threshold_module=None, unwrap=True,
                          voxel_sizes=None,
                          ax=None, fig=None, fw=3,
                          fig_title=None,
                          return_fig_ax=False):
    
#     if not np.any(np.iscomplex(obj)):
    if not (isinstance(obj[0,0,0], complex) or isinstance(obj[0,0,0], np.complex64) or isinstance(obj[0,0,0], np.complex128)) :
        # Function was called maybe by mistake on a real (non-complex) array
        plot_2D_slices_middle_one_array3D(obj, voxel_sizes=voxel_sizes,
                                          ax=ax, fig=fig, fw=fw, fig_title=fig_title)
        return
    
    if fig is None:
        fig, ax = plt.subplots(2,3, figsize=(3*fw, 2*fw))
        
    plot_2D_slices_middle_only_module(obj, ax=ax[0], fig=fig, voxel_sizes=voxel_sizes, crop=crop)  
    plot_2D_slices_middle_only_phase(obj, support=support, threshold_module=threshold_module,
                                     ax=ax[1], fig=fig,
                                     voxel_sizes=voxel_sizes, crop=crop, unwrap=unwrap)  
    
    ax[0,0].set_ylabel('module', fontsize=20)
    ax[1,0].set_ylabel('phase', fontsize=20)
    
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    
    fig.tight_layout()
        
    if return_fig_ax:
        return fig,ax
    else:
        return
# -----------------------------------------------------------------------------

def plot_3D_projections(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    alpha_mask: float = 0.3,
    ax=None,
    fig=None,
    fw: float = 4,
    fig_title: str | None = None,
    axes_labels: bool = False,
    colorbar: bool = False,
    log_scale: bool = True,
    log_threshold: bool = False,
    max_projection: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
):
    """
    Plot 3 orthogonal projections of a 3D volume.

    Projections are computed along each axis using either:
    - sum (default): physically consistent for diffraction intensity
    - max (optional): highlights peaks

    Parameters
    ----------
    data : np.ndarray
        3D array (D, H, W)
    mask : np.ndarray, optional
        Mask overlay (same shape as data)
    alpha_mask : float
        Transparency of mask overlay
    log_scale : bool
        Apply logarithmic scaling
    log_threshold : bool
        If True, apply manual log transform instead of LogNorm
    max_projection : bool
        Use max instead of sum projection
    fw : float
        Figure width scaling factor
    cmap : matplotlib colormap
    """
    data = _to_numpy(data)
    mask = _to_numpy(mask)

    if cmap is None:
        cmap = "plasma"

    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(3 * fw, fw))

    plots = []

    for n in range(3):

        proj = np.nanmax(data, axis=n) if max_projection else np.nansum(data, axis=n)

        if log_scale:
            if log_threshold:
                proj = np.log10(np.maximum(proj, 1e-12))
                im = ax[n].matshow(proj, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax[n].matshow(
                    proj, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax)
                )
        else:
            im = ax[n].matshow(proj, cmap=cmap, vmin=vmin, vmax=vmax)

        plots.append(im)

        if mask is not None:
            m = np.nanmean(mask, axis=n)
            m = (m != 0).astype(float)

            overlay = np.zeros((*m.shape, 4))
            overlay[..., 0] = m  # red channel
            overlay[..., 3] = alpha_mask * m

            ax[n].imshow(overlay)

    if axes_labels:
        ax[0].set_xlabel("detector horizontal")
        ax[0].set_ylabel("detector vertical")

        ax[1].set_xlabel("detector horizontal")
        ax[1].set_ylabel("rocking curve")

        ax[2].set_xlabel("detector vertical")
        ax[2].set_ylabel("rocking curve")

    if colorbar:
        for i, im in enumerate(plots):
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    if fig_title:
        fig.suptitle(fig_title)

    fig.tight_layout()
    return

# -----------------------------------------------------------------------------

def plot_module_histogram(obj,
                          crop=True,
                          bins=None, 
                          fig=None, ax=None):
    module, phase = get_cropped_module_phase(obj, crop=crop)
    module[module<.01*np.max(module)] = np.nan
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(6,4))
        
    if bins is None:
        if obj.ndim == 2:
            bins = 30
        if obj.ndim == 3:
            bins = 50
    n, bins, patches = ax.hist(module.flatten(), bins=bins)

    ax.set_xlabel('object module value', fontsize=15)
    ax.set_ylabel('number of pixels', fontsize=15)
    return


######################################################################################################################################
###########################            Plot slices in the middle of the array              ###########################################
###################################################################################################################################### 

def plot_2D_slices_middle_one_array3D(array,
                                      index=None,
                                      voxel_sizes=None, # voxel size in Angstrom
                                      add_colorbar=True, cbar_position='right', cbar_label=None,
                                 cmap='gray_r', norm=None,
                                 ax=None, fig=None,
                                 fw=3,
                                 fig_title = None, 
                                      xlabel=['nm','nm','nm'], ylabel=[None,None,None],
                                 alpha=1,
                                 vmin=None,vmax=None, aspect=None,
                                 symmetric_colorscale=False):
    
    if symmetric_colorscale:
        cmap='coolwarm'
    
    shape = array.shape
    
    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(3*fw,fw))
        
    if voxel_sizes is not None:
        voxel_sizes = .1*np.array(voxel_sizes) # Put the voxel_sizes in nanometers
        extent = [[0, array.shape[2]*voxel_sizes[2], 0, array.shape[1]*voxel_sizes[1]], 
                   [0, array.shape[2]*voxel_sizes[2], 0, array.shape[0]*voxel_sizes[0]],
                   [0, array.shape[1]*voxel_sizes[1], 0, array.shape[0]*voxel_sizes[0]]]
    else:
        extent= [None, None, None]
    
    im = []
    for n in range(3):
        s = [slice(None, None, None) for ii in range(3)]
        if index is None:
            s[n] = shape[n]//2
        else:
            s[n] = min(index,shape[n]-1)
        arr = array[tuple(s)]
        if symmetric_colorscale:
            vmax = np.nanmax(np.abs(arr))
            vmin = -vmax
        if hasattr(alpha, "__len__"):
            alpha_plot = np.copy(alpha[tuple(s)])
            # Need to make the plot twice to avoid a matplotlib bug
            im.append(ax[n].imshow(arr, cmap=cmap, vmin=vmin,vmax=vmax, extent=extent[n], norm=norm,aspect=aspect))
            ax[n].cla()
            ax[n].imshow(arr, cmap=cmap, alpha=alpha_plot, vmin=vmin,vmax=vmax, extent=extent[n], norm=norm,aspect=aspect)
        else:
            im.append(ax[n].imshow(arr, cmap=cmap, alpha=alpha, vmin=vmin,vmax=vmax, extent=extent[n], norm=norm,aspect=aspect))
    

    if add_colorbar:
        for ii, img in enumerate(im):
            divider = make_axes_locatable(ax[ii])
            cax = divider.append_axes(cbar_position, size='5%', pad=.05)
#             cax = divider.append_axes('right', size='10%', pad=pad_colorbar)
            cbar = fig.colorbar(img, cax=cax, orientation='vertical')
            if cbar_label is not None:
                cbar.ax.set_ylabel(cbar_label, rotation=270,fontsize=20*fw/4., labelpad=10*fw)
        
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20*fw/4.)
        
    if voxel_sizes is not None:
        for n in range(3):
            ax[n].set_xlabel(xlabel[n],fontsize=15*fw/4.)
            ax[n].set_ylabel(ylabel[n],fontsize=15*fw/4.)
            ax[n].xaxis.set_ticks_position('bottom')
            
    for axe in ax.flatten():
        axe.locator_params(nbins=4)
    
    fig.tight_layout()
        
    return

def plot_2D_slices_middle_only_module(obj,
                                      crop=False,
                                      voxel_sizes=None,
                                      cmap='gray_r', vmin=None,vmax=None,
                                  ax=None, fig=None, fw=3,
                                      fig_title=None,
                                  alpha=1):
    module, phase = get_cropped_module_phase(obj, unwrap=False, crop=crop)
    plot_2D_slices_middle_one_array3D(module, cmap=cmap, ax=ax, fig=fig, fw=fw, alpha=alpha, fig_title=fig_title,
                                      voxel_sizes=voxel_sizes,
                                      vmin=vmin, vmax=vmax)  
    return

def plot_2D_slices_middle_only_phase(obj,
                                     crop=False,
                                     threshold_module = None, support = None,
                                     voxel_sizes=None,
                                     cmap='hsv', vmin=None,vmax=None,
                                     unwrap=True,
                                     ax=None, fig=None, fw=3, 
                                    fig_title=None):
    
    module, phase = get_cropped_module_phase(obj,
                             threshold_module = threshold_module, support = support,
                             crop=crop, apply_fftshift=False, unwrap=unwrap)
    plot_2D_slices_middle_one_array3D(phase, cmap=cmap, vmin=vmin, vmax=vmax,
                                      ax=ax, fig=fig, fw=fw, fig_title=fig_title, voxel_sizes=voxel_sizes)  
    return

def plot_2D_slices_middle(obj,
                          crop=False,
                          support=None, threshold_module=None, unwrap=True,
                          voxel_sizes=None,
                          ax=None, fig=None, fw=3,
                          fig_title=None,
                          return_fig_ax=False):
    
#     if not np.any(np.iscomplex(obj)):
    if not (isinstance(obj[0,0,0], complex) or isinstance(obj[0,0,0], np.complex64) or isinstance(obj[0,0,0], np.complex128)) :
        # Function was called maybe by mistake on a real (non-complex) array
        plot_2D_slices_middle_one_array3D(obj, voxel_sizes=voxel_sizes,
                                          ax=ax, fig=fig, fw=fw, fig_title=fig_title)
        return
    
    if fig is None:
        fig, ax = plt.subplots(2,3, figsize=(3*fw, 2*fw))
        
    plot_2D_slices_middle_only_module(obj, ax=ax[0], fig=fig, voxel_sizes=voxel_sizes, crop=crop)  
    plot_2D_slices_middle_only_phase(obj, support=support, threshold_module=threshold_module,
                                     ax=ax[1], fig=fig,
                                     voxel_sizes=voxel_sizes, crop=crop, unwrap=unwrap)  
    
    ax[0,0].set_ylabel('module', fontsize=20)
    ax[1,0].set_ylabel('phase', fontsize=20)
    
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    
    fig.tight_layout()
        
    if return_fig_ax:
        return fig,ax
    else:
        return

def get_cropped_module_phase(
    obj: np.ndarray,
    threshold_module: float | None = None,
    support: np.ndarray | None = None,
    crop: bool = False,
    apply_fftshift: bool = False,
    unwrap: bool = True,
):
    """
    Extract amplitude and phase from a complex object.

    Includes optional:
    - cropping
    - support-based masking
    - phase unwrapping

    Returns
    -------
    module : np.ndarray
    phase : np.ndarray
    """
    obj = _to_numpy(obj)
    support = _to_numpy(support)

    if apply_fftshift:
        obj = np.fft.fftshift(obj)

    if crop:
        obj = crop_tensor_half_size(obj[None, ...])[0]
        if support is not None:
            support = crop_tensor_half_size(support[None, ...])[0]

    module = np.abs(obj)

    if support is None:
        if threshold_module is None:
            threshold_module = 0.3
        support = module >= threshold_module * np.nanmax(module)

    phase = np.angle(obj)

    if unwrap:
        mask = ~support.astype(bool)
        phase = np.ma.masked_array(phase, mask=mask)
        phase = unwrap_phase(phase).data

    phase[~support] = np.nan

    return module, phase


# -----------------------------------------------------------------------------

def plot_2D_slices_middle_and_histogram(obj,
                                        crop=False,
                                        support=None, threshold_module=None, unwrap=True,
                                        ax=None, fig=None, fw=3,
                                        fig_title=None,
                                        return_fig_ax=False):
    if fig is None:
        fig,ax = plt.subplots(3,3, figsize=(fw*3,fw*3))
    
    plot_module_histogram(obj, fig=fig, ax=ax[0,1], crop=crop)
    fig.delaxes(ax[0,0])
    fig.delaxes(ax[0,2])
    
    plot_2D_slices_middle(obj, 
                          support = support, threshold_module=threshold_module, unwrap=unwrap,
                          fig=fig, ax=ax[1:], fw=fw, crop=crop)
    
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    if return_fig_ax:
        return fig,ax
    else:
        return
    
def plot_2D_slices_middle_only_module_and_histogram(obj,
                                                    crop=False,
                                                    ax=None, fig=None, fw=3,
                                                    fig_title=None,
                                                    return_fig_ax=False):
    if fig is None:
        fig,ax = plt.subplots(2,3, figsize=(fw*3,fw*2))
    
    plot_module_histogram(obj, fig=fig, ax=ax[0,1], crop=crop)
    fig.delaxes(ax[0,0])
    fig.delaxes(ax[0,2])
    
    plot_2D_slices_middle_only_module(obj,
                          fig=fig, ax=ax[1], fw=fw, crop=crop)
    
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    
    if return_fig_ax:
        return fig,ax
    else:
        return



