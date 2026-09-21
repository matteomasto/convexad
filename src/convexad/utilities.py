
from .viz import * 
import numpy as np
import gc
import jax

def center_of_mass(array):
    cen = np.zeros(array.ndim)
    pos = np.indices(array.shape)
    proba = array/np.nansum(array)
    for n in range(array.ndim):
        cen[n] += np.nansum(pos[n]*proba)
    return cen

def center_of_mass_calculation_two_steps(data, 
                                         crop = 50, 
                                         plot=False):
    
    center = np.unravel_index(np.nanargmax(data), data.shape)

    cropping_dim = []
    for n in range(data.ndim):
        cropping_dim.append([max([0, int(center[n]-crop/2)]),  min(int(center[n]+crop//2), data.shape[n]-1)])


    s = [slice( cropping_dim[n][0],  cropping_dim[n][1] ) for n in range(data.ndim)]
    center2 = center_of_mass(data[tuple(s)])

    center = [int(round(cropping_dim[n][0]+center2[n])) for n in range(data.ndim)]
    
    if plot:
        if data.ndim==3:
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            plot_3D_projections(data, fig=fig, ax=ax)
            ax[0].scatter(center[2], center[2], color='w')
            ax[1].scatter(center[2], center[0], color='w')
            ax[2].scatter(center[1], center[0], color='w')
        if data.ndim==2:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(np.log(data), cmap='plasma', vmin=1)
            plt.colorbar()
            plt.scatter(center[1], center[0], color='w')
    return center


def center_the_center_of_mass(data,
                              qx=None,qy=None,qz=None,
                              standard_com=False,
                              plot=False, vmin=None,
                              return_offsets=False,
                              cmap='plasma', norm=None,
                              scatter_color='g', scatter_size=10):
    '''
    Center the center of mass of a 3D matrix 
    I use this to center the Bragg peak after the small random shift I did in the function "Createqxqyqz"
    '''
    shape = data.shape
    
    data[~np.isfinite(data)] = 0
    
    # Calculate where is the center of mass
    if standard_com:
        com = center_of_mass(data)
    else:
        com = center_of_mass_calculation_two_steps(data)
    
    # Calculate what's the offset to put back the center of mass at the middle of the 3D matrix
    offset = [int(np.rint(shape[n] / 2.0 - com[n])) for n in range(len(shape)) ]

    # Put back the center of mass to the middle of the matrix
    data_cen = np.roll(data, offset, axis=range(len(shape)))  
    if qx is not None:
        qx = np.roll(qx, offset, axis=range(len(shape)))  
        qy = np.roll(qy, offset, axis=range(len(shape)))  
        qz = np.roll(qz, offset, axis=range(len(shape)))  
    if plot:
        if len(shape)==2:
            fig,ax = plt.subplots(1,2, figsize=(8,4))
            ax[0].imshow(data, cmap=cmap, vmin=vmin, norm=norm)
            ax[0].scatter(com[1],com[0], c=scatter_color, s=scatter_size)
            ax[1].imshow(data_cen, cmap=cmap, vmin=vmin, norm=norm)
#         if len(shape)==3:
#             plot_3D_projections(data, fig_title='original data')
#             plot_3D_projections(data_cen, fig_title='centered data')
    
    if return_offsets:
        return data_cen, offset
    else:
        if qx is not None:
            return data_cen, qx, qy, qz
        else:
            return data_cen

def center_object(obj, 
                  standard_com=True,
                  support=None):
    module = np.abs(obj)
    
    module_cen, offset = center_the_center_of_mass(module, return_offsets=True, standard_com=standard_com)
    
    # I make the centering twice, I use a cropping for the second one
    shape = obj.shape
    module_cen = crop_array_half_size(module_cen)
    module_cen2, offset2 = center_the_center_of_mass(module_cen, return_offsets=True,standard_com=standard_com)

    total_offset = np.array(offset)+np.array(offset2)

    obj_cen = np.roll(obj, total_offset, axis=range(len(shape))) 
    
    if support is not None:
        support_cen = np.roll(support, total_offset, axis=range(len(shape)))
        return obj_cen, support_cen
    else: 
        return obj_cen

def crop_array_half_size(array):
    """
    Crop the array around its center to half its size in every dimension.

    For each dimension N, the output size is ceil(N / 2).
    Works for any number of dimensions.
    """
    shape = array.shape

    slices = []
    for n in shape:
        target = (n + 1) // 2  # ceil(n / 2)
        start = (n - target) // 2
        stop = start + target
        slices.append(slice(start, stop))

    return array[tuple(slices)]

def free_gpu(*names, namespace=None):
    """Delete the given variable names from the given namespace (default: globals())
    and force JAX to actually release the memory."""
    ns = namespace if namespace is not None else globals()
    for name in names:
        ns.pop(name, None)
    gc.collect()
    jax.clear_caches()