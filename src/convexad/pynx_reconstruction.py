import numpy as np
import pylab as plt
from scipy.fft import ifftn, fftn, fftshift, ifftshift
from pynx.cdi import * # Might take a bit of time to import
import inspect # To check if the tv is available

from .viz import *
from .utilities import *

my_cmap = 'plasma'


######################################################################################################################################
################################                      Dummy file_dict                       ##########################################
######################################################################################################################################

def dummy_file_dict(savename='dummy'):
    file_dict = {}
    file_dict['orthogonalization'] = False
    file_dict['h5file'] = savename
    file_dict['scan_nb'] = 0
    file_dict['savename_add_string'] = ''
    file_dict['sample'] = 'dummy'
    return file_dict

######################################################################################################################################
################################            Load preprocessed diffraction data              ##########################################
######################################################################################################################################
    
from matplotlib.colors import LogNorm
def load_diffraction_data_Q_space(preprocessed_datapath,
                                  vmin=1,
                                  plot=False):
    file = np.load(preprocessed_datapath, allow_pickle=True)
    data = file['data']

    if plot:
        if len(data.shape)==2:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(data, cmap='gray',norm=LogNorm(vmin=vmin))
            plt.colorbar()
            
        if len(data.shape)==3:
            plot_3D_projections(data)
   
    mask = file['mask']
    
    # load dictionary
    file_dict = {key : file[key] for key in file.files}
    file_dict['preprocessed_datapath'] = preprocessed_datapath
    file_dict.pop('data', None)
    file_dict.pop('mask', None)
    
    return data, mask, file_dict

######################################################################################################################################
###################################            Random object initialization             ##############################################
######################################################################################################################################

def CreateRandomInitialObject(shape, amin=0, amax=80, phirange=np.pi/2.):
    a = np.random.uniform(amin, amax, shape)
    phi = np.random.uniform(0, phirange, shape)
    obj_init = a*np.exp(1.0j*phi) 
    return obj_init

######################################################################################################################################
###################################             Create support from object              ##############################################
######################################################################################################################################

import scipy
def EB_custom_support_from_object(obj,
                               sigma=3, threshold=.1,
                               plot=False, alpha=.3):
    module = np.abs(obj)
    module_gauss = scipy.ndimage.gaussian_filter(module,sigma=sigma)
    
    support = np.zeros(module.shape)
    support[module_gauss>threshold*np.max(module_gauss)] = 1

    if plot:
        fig,ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].imshow(module, cmap='gray_r')
        ax[1].imshow(module, cmap='gray_r')
        ax[1].imshow( np.dstack([support, np.zeros(support.shape), np.zeros(support.shape), alpha*support]))
        ax[0].set_title('module (gray)', fontsize=15)
        ax[1].set_title('module (gray)\nsupport (transparent red)', fontsize=15)
    return support

######################################################################################################################################
###########################            CDI reconstruction (works well for 2D data)             #######################################
######################################################################################################################################

def default_cdi_parameters():
    params = {}
    params['center_data'] = False # center data during reconstruction
    params['show_cdi'] = 300# None#10000#300
    # params['algo_string'] = 'HIO_600 RAAR_1000 ER_200 HIO_500'
    params['algo_string'] = 'HIO_400 RAAR_1000 ER_300'
    params['support_update'] = 20
    params['fix_support'] = False
    params['init_psf'] = True
    params['update_psf'] = 20

    params['obj_init'] = None
    params['support_init'] = 'gauss_conv'
    params['mask'] = None
    params['cdi'] = None

    # params['support_threshold_relative_min'] = .23
    # params['support_threshold_relative_max'] = .3
    params['support_threshold_relative'] = None
#     params['support_threshold_relative_min'] = .2
    params['support_threshold_relative_min'] = .1
    params['support_threshold_relative_max'] = .4
    params['support_smooth_width'] = (2,1,600)
    params['post_expand'] = (1,-2,1)
    params['support_update_method'] = 'rms'
    params['update_border_n'] = 0

    params['plot_result'] = True
    params['verbose'] = True
    params['return_cdi'] = False

    params['compute_free_llk'] = True
    params['calc_llk'] = None # Don't calculate llk all the time
    
    params['filling_object'] = False
    
    params['tv'] = None #0.1
    
    return params

import warnings
def CDI_one_reconstruction(data, params,
                           plot_result=True):


    '''
    :support_init: 'gauss_conv' or 'autocorrelation'. gauss_conv only works if the obj_init_list is given
    :obj_init: careful, should not be fftshifted
    '''
    
    if np.any(np.array(data.shape) %2 != 0):
        params['center_data'] = False
        warnings.warn("Centering removed. Centering is dangerous with odd array dimension. put_back_centering_ramp function might need changes.", UserWarning)
              
    if params['center_data']:
        data, centering_offsets = center_the_center_of_mass(data, return_offsets=True)
        if params['mask'] is not None:
            params['mask'] = np.roll(params['mask'], centering_offsets, axis=range(len(params['mask'].shape)))  
        print('centering_offsets : ', centering_offsets)
    
    if type(params['support_init']) == np.ndarray:
        support = params['support_init']
        print('using the support array given by user')
    elif type(params['support_init']) == str:
        if params['support_init']=='gauss_conv' and params['obj_init'] is not None:
            if params['verbose']:
                print('using gaussian convoluted and threshold as support')
            support = EB_custom_support_from_object(params['obj_init'], plot=False)
        else :
            support = None # Autocorrelation support will be used
    else:
        raise ValueError('something is wrong in params[\'support_init\']')
    
    if params['obj_init'] is None:
        obj_init = CreateRandomInitialObject(data.shape)
        if type(params['support_init'])==str:
            if params['verbose']:
                print('Using autocorrelation as an initial support. If you\'re not happy, give either a support or an obj_init')
            params['support_init'] = 'autocorrelation'
    else:
        obj_init = np.copy(params['obj_init'])
    obj_init = fftshift(obj_init)

    if support is not None:
        support = fftshift(support)
        
    if params['mask'] is not None:
        print('using mask')
        mask = fftshift(params['mask'])
    else:
        mask = np.zeros(data.shape)
        
    if params['cdi'] is None:
        cdi = CDI(fftshift(data), obj=obj_init, support=support, mask=mask)
    else:
        cdi = params['cdi']
        cdi.set_obj(params['obj_init'])
        cdi.set_support(support)
                
    cdi = ScaleObj() * cdi
#     cdi = InitPSF()*cdi
    if params['init_psf']:
        cdi = InitPSF(fwhm=.5, eta=0.05)*cdi # Used to take partial coherence into account    

    if type(params['support_init'])==str: 
        if params['support_init']=='autocorrelation':
            if params['verbose']:
                print('using autocorrelation as support')
            cdi = AutoCorrelationSupport(threshold=0.1) * cdi
        
    if params['compute_free_llk']:
        cdi.init_free_pixels() # Used to compute the free log likehood. In the end it doesn't work very well
        
    if params['support_threshold_relative'] is None:
        support_threshold_relative = params['support_threshold_relative_min'] \
                                    + np.random.rand()\
                                    *(params['support_threshold_relative_max']-params['support_threshold_relative_min'])
        
    else:
        support_threshold_relative = np.copy(params['support_threshold_relative'])
#     sup = SupportUpdate(threshold_relative=support_threshold_relative, smooth_width=params['support_smooth_width'], 
#                 force_shrink=False,method='max', post_expand=None)
    sup = SupportUpdate(threshold_relative=support_threshold_relative, smooth_width=params['support_smooth_width'], 
                force_shrink=False, post_expand=params['post_expand'], 
                       method=params['support_update_method'],
                       update_border_n = params['update_border_n'])
    
    kwarg_dict = {}
    if params['tv'] is not None:
        # Safety check 
        sig = inspect.signature(ER)
        if not 'tv' in sig.parameters:
            raise ValueError('tv is not available in this pynx environnment')
        kwarg_dict['tv'] = params['tv']
        print(f"tv regularization at : {kwarg_dict['tv']}")

    if params['show_cdi'] is not None:
        plt.figure()
    
    # Reconstruction algortihm
    for algo in params['algo_string'].split():
        method = algo.split('_')[0]
        iterations = int(algo.split('_')[1])
        
        if params['support_update']==False or params['support_update']==None:
            if method == 'HIO':
                cdi =  HIO(beta=0.9, calc_llk=params['calc_llk'], show_cdi=params['show_cdi'],
                           update_psf=params['update_psf'], **kwarg_dict)**iterations* cdi 
            if method == 'DetwinHIO':
                cdi = DetwinHIO(beta=0.9)**iterations * cdi 
            if method == 'RAAR' :
                cdi = RAAR(beta=0.9, calc_llk=params['calc_llk'], show_cdi=params['show_cdi'],
                           update_psf=params['update_psf'], **kwarg_dict)**iterations* cdi
            if method == 'DetwinRAAR':
                cdi = DetwinRAAR(beta=0.9)**iterations * cdi 
            if method == 'ER' :
                cdi = ER(calc_llk=params['calc_llk'], show_cdi=params['show_cdi'],
                         update_psf=params['update_psf'], **kwarg_dict) **iterations *cdi

        else:   
            if method == 'HIO':
                cdi = (sup * HIO(beta=0.9, calc_llk=params['calc_llk'], show_cdi=params['show_cdi'], 
                                 update_psf=params['update_psf'], **kwarg_dict)**params['support_update'])**(iterations//params['support_update'])* cdi 
            if method == 'DetwinHIO':
                cdi = DetwinHIO(beta=0.9)**iterations * cdi 
            if method == 'RAAR' :
                cdi = (sup * RAAR(beta=0.9, calc_llk=params['calc_llk'], show_cdi=params['show_cdi'], 
                                  update_psf=params['update_psf'], **kwarg_dict)**params['support_update'])**(iterations//params['support_update'])* cdi
            if method == 'DetwinRAAR':
                cdi = DetwinRAAR(beta=0.9)**iterations * cdi 
            if method == 'ER' :
                cdi = (sup * ER(calc_llk=params['calc_llk'], show_cdi=params['show_cdi'], 
                                update_psf=params['update_psf'], **kwarg_dict) ** params['support_update'])**(iterations//params['support_update']) *cdi

    if params['calc_llk']==0:
        # I fucking hate this line. I'm struggling to calculate the llk at the end.
        # My dirty fix is to add one step of ER and calculate llk inside.
        cdi = (sup * ER(calc_llk=1) )* cdi
        
    if params['compute_free_llk']:
        llk = cdi.get_llk()[3] #  keep free poisson log likelihood
    else:
        llk = cdi.get_llk()[0] #  keep poisson log likelihood
    
    obj = fftshift(cdi.get_obj())
    support = fftshift(cdi.get_support())
    obj, support = center_object(obj, support=support)
    
    if params['center_data']:
        print('centering ramp is added back at the end of the reconstruction')
        obj = put_back_centering_ramp(obj, centering_offsets)
    
    if params['plot_result']:
        if obj.ndim ==3:
            plot_2D_slices_middle_and_histogram(obj, support=support, threshold_module = support_threshold_relative)
        if obj.ndim ==2:
            plot_object_module_phase_2d(obj)
        
    return_dict = {}
    return_dict['support_threshold_relative'] = support_threshold_relative
    if params['return_cdi']:
        return_dict['cdi'] = cdi

    return obj, llk, support, return_dict
    

######################################################################################################################################
###########################                Put back ramp due to the centering                   ######################################
######################################################################################################################################

def put_back_centering_ramp(obj, centering_offsets):
    Fexp = ifftshift(fftn(fftshift(obj)))
    Fexp = np.roll(Fexp, -np.array(centering_offsets), axis=range(len(Fexp.shape)))   
    obj = fftshift(ifftn(ifftshift(Fexp)))
    return obj
