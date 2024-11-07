# -*- coding: utf-8 -*-
import os, shutil, logging
import numpy as np
import nibabel as nib
import json 
from dipy.reconst import dti
from dipy.core.gradients import gradient_table
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma, piesno
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate 
from .utils import make_dir, make_temp_dir, ExecCommand, DiciphrException, which
from .nifti_utils import ( read_nifti, write_nifti, nifti_image, erode_image,
                is_valid_dwi, read_dwi, write_dwi, concatenate_niftis )

##############################################
####### Bvals/bvecs, DWI manipulation ########
##############################################
    
def round_bvals(bvals,nearest='hundred'):
    '''Round a bvals array to the specified significance.
    
    Parameters
    ----------
    bvals : numpy.ndarary
        The bvals array
    nearest : Optional[str]
        One of 'thousand','hundred','ten','one'
    
    '''
    nearest_dict = {'one':1.,
                    'ten':10.,
                    'hundred':100.,
                    'thousand':1000.}
    factor = nearest_dict[nearest]
    bvals = bvals / factor
    bvals = np.round(bvals)
    bvals = bvals * factor
    return bvals
    
def number_unique_bvalues(bvals):
    '''Get the number of nonzero unique b-values in a bvals array.
    
    Can be used to determine if round_bvals is necessary, or 
    if a diffusion weighted sequence cannot be rounded to one uniform value.
    
    Parameters
    ----------
    bvals : numpy.ndarray
        The bvals array
        
    Returns
    -------
    int
        Number of nonzero unique b-values
    '''
    return np.size(np.unique(bvals[bvals > 0]))
    
def number_unique_gradients(bvecs):
    '''Returns number of unique gradient directions in a bvecs set.'''
    class BvecsSet:
        def __init__(self):
            self.data=[]
        def __len__(self):
            return len(self.data)
        def add(self, vec):
            if np.linalg.norm(vec) > 1e-4: #skip b0
                if len(self.data) == 0: 
                    self.data.append(vec)
                else: 
                    _append=True
                    for other_vec in self.data:
                        if angle_between(vec, other_vec) < 1e-4: 
                            _append=False
                    if _append:
                        self.data.append(vec)
    n=bvecs.shape[1]
    inds=range(n)
    unique_set = BvecsSet()
    for vec in bvecs.transpose():
        unique_set.add(vec)
    return len(unique_set)

def _affines_match(aff1, aff2):
    '''Returns true if two affines match within an error of 1e-4.'''
    origin_distance = np.sqrt((aff1[0,3]-aff2[0,3])**2+(aff1[1,3]-aff2[1,3])**2+(aff1[2,3]-aff2[2,3])**2)
    origin_norm = np.sqrt((aff1[0,3])**2+(aff1[1,3])**2+(aff1[2,3])**2)
    origin_same = (origin_distance / origin_norm) < 1e-4
    rotation_same=np.sum(np.abs(aff1[0:3,0:3] - aff2[0:3,0:3])) < 1e-4
    return rotation_same and origin_same

def concatenate_dwis(*args):
    '''Concatenate two or more dwi images into one.
    
    Positional arguments are input as tuples, (dwi_im, bvals, bvecs),
    dwi_im is nibabel.Nifti1Image. bvals, bvecs are numpy.ndarray.
    
    Example usage:
        dwi1 = read_dwi('dwi_file1.nii.gz') #returns dwi_im, bvals, bvecs tuple
        dwi2 = read_dwi('dwi_file2.nii.gz')
        dwi_cat = concatenate_dwis(dwi1,dwi2)
    
    Parameters
    ----------
    \*args : 
        Tuples (dwi_im, bvals, bvecs)
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.concatenate_dwis')
    dwi_im_list = [a[0] for a in args]
    bvals_list = [a[1] for a in args]
    bvecs_list = [a[2] for a in args]
    dwi_data_list = [a.get_data() for a in dwi_im_list]
    
    dwi_data_out = np.concatenate(dwi_data_list, axis=3)
    bvals_out = np.concatenate(bvals_list, axis=1)
    bvecs_out = np.concatenate(bvecs_list, axis=1)
    
    affines=[a.affine for a in dwi_im_list]
    affines_match = True
    for idx in range(len(affines)-1):
        affines_match = affines_match and _affines_match(affines[idx], affines[idx+1])
    aff = affines[0]
    hdr=dwi_im_list[0].header 
    
    dwi_im_out = nifti_image(dwi_data_out, aff, hdr)
    is_valid_dwi(dwi_im_out, bvals_out, bvecs_out, True)
    return dwi_im_out, bvals_out, bvecs_out
    
def join_dwi(*nifti_images):
    '''Join dwi volumes into one 4D images, along the time axis 
    
    Parameters
    ----------
    nifti_images : list
        a list of nibabel.Nifti1Image objects
    
    Returns
    -------
    nib.Nifti1Image
        A 4D image 
    '''
    affine = nifti_images[0].affine 
    datas = [ ] 
    for im in nifti_images:
        dat = im.get_data()
        if len(dat.shape) == 3:
            dat = dat[...,np.newaxis]
        elif len(dat.shape) != 4:
            raise DiciphrException('join_dwi expects inputs to be 3D or 4D!')
        datas.append(dat)
    dwi_data = np.concatenate(datas, axis=3)
    dwi_im = nifti_image(dwi_data, affine)
    return dwi_im
    
def split_dwi(dwi_im):
    '''Split dwi images into volumes, i.e. along the time axis 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI Image
    
    Returns
    -------
    list
        A list of nibabel.Nifti1Image objects 
    '''
    dwi_data = dwi_im.get_data()
    dwi_affine = dwi_im.affine
    ret = []
    nt = dwi_data.shape[-1]
    for i in range(nt):
        dat = dwi_data[...,i]
        ret.append(nifti_image(np.squeeze(dat), dwi_affine))
    return ret

def remove_dwi_gradients(dwi_im,bvals,bvecs,list_of_indices):
    '''Remove gradient images and associated bvectors from a DWI image.
    
    Given a dwi image, bvals, bvecs and a list of indices, 
    will remove subimages at those indices from the dwi image volume 
    as well as the associated elements from the bval and bvec arrays.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
    list_of_indices : list
        The indices to remove.
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    is_valid_dwi(dwi_im, bvals, bvecs, True) # raises if not
    data=dwi_im.get_data()
    if len(list_of_indices) < 1:
        logging.warning('Asked to remove an empty list of gradient images. Returning original DWI')
        return (dwi_im, bvals, bvecs )
    if max(list_of_indices) >= data.shape[-1]:
        raise DiciphrException('Cannot remove dwi gradient. Index out of range!')
    
    bvals_out=bvals.reshape((1,np.size(bvals)))
    bvecs_out=bvecs.copy()
    # are next 2 lines needed? probably not 
    if not (np.size(bvals_out) == np.size(bvecs_out)/3 and np.size(bvals_out) == data.shape[-1]): 
        raise DiciphrException('Bval, bvec and data do not encode the same number of gradients!')
    list_of_indices=sorted(list_of_indices,reverse=True)
    for ind in list_of_indices:
        data = np.delete(data,ind,axis=3)
        bvals_out=np.delete(bvals_out,ind,axis=1)
        bvecs_out=np.delete(bvecs_out,ind,axis=1)
    hdr=dwi_im.header
    hdr['dim'][4] -= len(list_of_indices)
    dwi_image_out=nifti_image(data,dwi_im.affine,header=hdr)
    return (dwi_image_out,bvals_out,bvecs_out)    
    
def extract_b0(dwi_im, bvals, bvecs=None, first=False, average=True):
    '''Extract the average B0 from a dwi volume. 
    
    bvals are used, and bvecs are not, but the argument is provided, 
    to preserve usage relative to other dwi functions in this module. 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarary
        The bvals array
    bvecs : Optional[numpy.ndarray]
        The bvecs array. Not used
    first : Optional[bool]
        If true, return the first encountered B0. 
    average: Optional[bool]
        If true ( default ) averages the B0s, else returns a 4D volume of B0 images.
    Returns
    -------
    nibabel.Nifti1Image
        The average B0 image.
    '''
    logging.debug('diciphr.diffusion.extract_b0')
    data = dwi_im.get_data()
    if first:
        indy = np.where(bvals ==0)[1][0]
        b0 = data[...,indy]
    else:
        indy = np.where(bvals == 0)[1]
        if average:
            b0 = np.mean(data[...,indy],axis=-1)
        else:
            b0 = data[...,indy]
    b0_im=nifti_image(b0, dwi_im.affine, dwi_im.header)
    return b0_im

def extract_gaussian_shells(dwi_im, bvals, bvecs):
    if np.max(bvals) > 1500 or np.min(bvals[bvals>0]) < 500:
        logging.info("B-values outside the Gaussian range 500-1500 detected.")
        bvals_gaussian = list(np.unique(bvals[np.logical_or(bvals >= 500, bvals <=1500)]))
        if len(bvals_gaussian) == 0 :
            logging.info("No b-values between 500 and 1500 detected. Returning original DWI")
            return dwi_im, bvals, bvecs 
        else:
            shells = [0] + bvals_gaussian
            logging.info("Extracted b-values: {}".format(shells))
            return extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, shells)
    else:
        logging.info("B-values already inside the Gaussian range 500-1500.")
        return dwi_im, bvals, bvecs 

def extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, target_bvalues, error=20):
    ''' Extract the gradients from a diffusion weighted image that are within error of target bvalues.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI image
    bvals : numpy.ndarray
        The 1xN bvals array
    bvecs : numpy.ndarray
        The 3xN bvecs array
    target_bvalues : list 
        A list of bvalues to keep.
    error : Optional[int]
        Keep gradients with bvalues within this much of the target. 
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.extract_shells_from_multishell_dwi')
    if not hasattr(target_bvalues, '__getitem__'):
        target_bvalues=[target_bvalues]
    indices_to_remove = set(range(np.size(bvals)))
    for bvalue in target_bvalues:
        indices = set(np.where(np.abs(bvals - bvalue) > error)[1])
        indices_to_remove.intersection_update(indices)
    indices_to_remove = sorted(list(indices_to_remove))
    if len(indices_to_remove) == 0:
        logging.info('No indices to remove. Returning input DWI data.')
        return dwi_im, bvals, bvecs
    return remove_dwi_gradients(dwi_im, bvals, bvecs, indices_to_remove)

##############################################
#####     DWI Intensity manipulation      ####
##### Masking, bias, denoising, normalize ####
##############################################

def mask_dwi(dwi_im, mask_im):
    '''Multiples each image in a DWI by mask and returns a Nifti1Image object. 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    mask_im : nibabel.Nifti1Image
        The mask image
    Returns
    -------
    nibabel.Nifti1Image
        Masked DWI image (no bval, bvec)
    '''
    dwi_data=dwi_im.get_data()
    mask_data = mask_im.get_data().astype(int)
    mask_data = mask_data[...,np.newaxis]
    dwi_masked_data = dwi_data * mask_data
    dwi_masked_im = nifti_image(dwi_masked_data.astype(dwi_im.get_data_dtype()), dwi_im.affine, dwi_im.header)
    return dwi_masked_im
     
def n4_bias_correct_dwi(dwi_im, bvals, bvecs=None, mask_im=None, iterations=[50,50,50,50], threshold=0.001, spline_distance=150, mesh_resolution=3, return_field=False):
    '''
    Runs ANTS N4BiasFieldCorrection on a DWI image.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    mask_im: nibabel.Nifti1Image
        A mask NiFTI image 
    bvals : numpy.ndarary
        The bvals array
    bvecs : Optional[numpy.ndarray]
        The bvecs array. Not used
        
    Returns
    -------
    nibabel.Nifti1Image
        The corrected DWI image (not including bvals/bvecs)
    '''
    n4_cmd = which('N4BiasFieldCorrection')
    tmpdir = make_temp_dir(prefix='n4BiasDwi')
    b0_filename = os.path.join(tmpdir, 'b0.nii')
    mask_filename = os.path.join(tmpdir, 'mask.nii')
    b0_n4_filename = os.path.join(tmpdir, 'corrected.nii')
    bias_filename = os.path.join(tmpdir, 'bias.nii')
    try:
        b0_im = extract_b0(dwi_im, bvals)
        b0_im.to_filename(b0_filename)
        if mask_im is not None:
            mask_im.to_filename(mask_filename)
        else:
            mask_im = bet2_mask_nifti(b0_im)
            mask_im.to_filename(mask_filename)
        cmd=[n4_cmd,
                '-i',b0_filename,
                '-d','3',
                '-x',mask_filename,
                '-o','[' + b0_n4_filename + ',' + bias_filename + ']',
                '-b','[{s},{m}]'.format(s=spline_distance, m=mesh_resolution),
                '-c','[{i},{t}]'.format(i='x'.join(map(str,iterations)),t=threshold),
        ]
        ExecCommand(cmd).run()
        bias_data = nib.load(bias_filename).get_data().astype(float)
        bias_mean = np.mean(bias_data) # makes the mean of the output image same as input 
        dwi_data = dwi_im.get_data()
        dwi_corrected_data = dwi_data * bias_mean / bias_data[...,None]
        dwi_corrected_im = nifti_image(dwi_corrected_data.astype(dwi_im.get_data_dtype()), dwi_im.affine)
        bias_im = nifti_image(bias_data/bias_mean, dwi_im.affine)
    finally:
        shutil.rmtree(tmpdir)
    if return_field:
        return dwi_corrected_im, bias_im
    else:
        return dwi_corrected_im

def normalize_dwi(dwi_im, bvals, bvecs, reference_value=1000.0, wm_im=None, mask_im=None):
    '''Normalize a DWI by adjusting the median B0 signal in WM to a reference value. 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The 4D DWI Nifti image object.
    bvals : numpy.ndarray 
        A bvals array of shape (N,)
    bvecs : numpy.ndarray
        A bvecs array of shape (3,N)
    reference_value : Optional[float]
        A number to set 
    wm_im : Optional[nibabel.Nifti1Image]
        A mask of WM voxels. If not provided, a tensor will be fit and FA thresholded at 0.30 
    mask_im : Optional[nibabel.Nifti1Image]
        The brain mask for tensor fit, if wm_im is not provided. Will run BET2 on b0 if not provided.     
    
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.normalize_dwi')
    b0_im = extract_b0(dwi_im, bvals)
    affine = dwi_im.affine
    b0_data = b0_im.get_data()
    if wm_im is None:
        fa_wm_threshold = 0.30
        # threshold FA to get WM mask 
        if mask_im is None:
            mask_im = bet2_mask_nifti(b0_im, erode_iterations=1)
        tensor_im, fa_im, tr_im = estimate_tensor(dwi_im, mask_im, bvals, bvecs)
        wm_mask = fa_im.get_data() > fa_wm_threshold
    else:
        wm_mask = wm_im.get_data() > 0 
    b0_wm_median_value = np.median(b0_data[wm_mask])
    scale = reference_value / b0_wm_median_value
    dwi_data_scaled = dwi_im.get_data() * scale
    dwi_out_im = nifti_image(dwi_data_scaled, affine)
    return dwi_out_im, bvals, bvecs 

def lpca_denoise(dwi_im, bvals, bvecs, return_diff=False, smooth=3, tau_factor=2.3, patch_radius=2):
    logging.debug('diciphr.diffusion.lpca_denoise_dipy')
    data = dwi_im.get_data()
    hdr = dwi_im.header 
    affine = dwi_im.affine
    gtab = gradient_table(bvals.flatten(), bvecs.T)
    logging.info('Estimate sigma for pca')
    sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=smooth)
    logging.info('Local PCA denoising')
    denoised_arr = localpca(data, sigma, tau_factor=tau_factor, patch_radius=patch_radius)
    logging.info('Local PCA done.')
    dwi_denoised_im = nifti_image(denoised_arr, affine, hdr)
    if return_diff:
        diff_im = nifti_image(dwi_im.get_data() - dwi_denoised_im.get_data(), affine, hdr)
        return dwi_denoised_im, diff_im
    else:
        return dwi_denoised_im 
    
def estimate_noise(dwi_im, piesno=False, N=0):
    '''Calculate sigma from the dwi image, with module dipy.denoise.noise_estimate 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    piesno : Optional[bool]
        Runs dipy's piesno noise estimation. Warning may produce zeros in sigma mask.
    N : Optional[int]
        Number send to dipy's noise module... 
    verbose : Optional[int]
        Print info message about noise standard deviation
    Returns
    -------
    numpy.ndarray
        sigma
    '''
    if piesno:
        sigma = np.zeros(dwi_im.shape[:3], dtype=np.float32)
        for idx in range(dwi_im.shape[-2]):
            logging.info("Now processing slice {} out of {}".format(idx+1,dwi_im.shape[-2]))
            sigma[...,idx] = piesno(dwi_im.get_data()[...,idx,:], N=N, return_mask=False)
        logging.debug("Noise standard deviation from piesno is {}".format(sigma[0,0,:]))
    else:
        sigma = estimate_sigma(dwi_im.get_data(), N=N)
        #sigma = np.median(sigma)  #added 2016-02-08
        logging.info("Noise standard deviation from estimate_sigma is {}".format(sigma))
        #sigma = np.ones(dwi_im.shape[:3])*sigma
    return sigma
    
def run_nlmeans(dwi_im, sigma, mask_im=None, N=1):
    '''Calculate sigma from the dwi image, with module dipy.denoise.noise_estimate 
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    sigma : numpy.ndarray or float
        Result from estimate_noise
    mask_im : nibabel.Nifti1Image
        The mask image
    N : Optional[int]
        Number send to dipy's noise module... 
    Returns
    -------
    nibabel.Nifti1Image
        Denoised DWI image (no bval, bvec)
    '''
    rician=(N > 0)
    mask=None
    if mask_im is not None:
        mask=mask_im.get_data()
    dwi_data = dwi_im.get_data()
    logging.debug("Running nlmeans with rician {}".format(rician))
    dwi_denoised_data = nlmeans(dwi_data, sigma, mask=mask, rician=rician)
    dwi_denoised_data = dwi_denoised_data.astype(dwi_data.dtype)
    dwi_denoised_im = nifti_image(dwi_denoised_data,dwi_im.affine)
    return dwi_denoised_im
    
def adjust_b0_problem_voxels(dwi_im, bvals, lmin=1e-9):
    # Adjust B0 voxels that are too low based on tissue properties 
    # i.e. such that the least signal attenuation has a diffusivity of lmin
    dwi = dwi_im.get_data()
    b0 = dwi[...,bvals.flatten() == 0]
    wtd = dwi[...,bvals.flatten() > 0]
    wtdmax = np.max(wtd, axis=-1)
    b = np.min(bvals[bvals>0])
    bmin = wtdmax / np.exp(-1*b*lmin)
    bmin = np.tile(bmin[...,None], b0.shape[-1])
    b0[b0 < bmin] = bmin[b0 < bmin] 
    dwi[..., bvals.flatten() == 0] = b0
    return nifti_image(dwi, dwi_im.affine, dwi_im.header)

##############################################
###############     Tensors     ##############
##############################################

def is_tensor(nifti_im):
    '''Reads intent_code from the header and returns true if image is a tensor'''
    return nifti_im.header['intent_code']==1005 and len(nifti_im.shape) == 5 and nifti_im.shape[4] == 6
   
def estimate_tensor(dwi_im, mask_im, bvals, bvecs, fit_method='WLS'):
    '''Estimate the tensor image using dipy.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    mask_im: nibabel.Nifti1Image
        A mask NiFTI image 
    bvals : numpy.ndarary
        The bvals array
    bvecs : Optional[numpy.ndarray]
        The bvecs array. Not used
    fit_method : Optional[str]
        The fitting method, one of "WLS","OLS","NLLS"
        
    Returns
    -------
    tuple
        (tensor_im, fa_im, tr_im)  tuple of nibabel.Nifti1Image
    '''
    gtab = gradient_table(bvals.flatten(), bvecs.transpose()) #dipy is transposed
    aff = dwi_im.affine
    hdr = dwi_im.header.copy()
    hdr.set_xyzt_units(hdr.get_xyzt_units()[0],0) # remove seconds from header
    
    tenmodel = dti.TensorModel(gtab, fit_method=fit_method)
    tenfit = tenmodel.fit(dwi_im.get_data(), mask=(mask_im.get_data() > 0)) #saves some time
    
    tensor_data = tenfit.lower_triangular().astype('float32')
    tensor_data = tensor_data[...,np.newaxis,:]
    tensor_data_mask = mask_im.get_data().reshape(list(mask_im.shape) + [1,1])
    tensor_data = tensor_data * tensor_data_mask
    tensor_im = nifti_image(tensor_data, aff, hdr, intent=1005)
    
    fa_data = (tenfit.fa * mask_im.get_data()).astype('float32')
    fa_im = nifti_image(fa_data,aff,hdr, cal_min=0.0, cal_max=1.0)
    
    tr_data = (tenfit.trace * mask_im.get_data()).astype('float32')
    tr_im = nifti_image(tr_data,aff,hdr)
    
    return tensor_im, fa_im, tr_im 
    
def estimate_tensor_restore(dwi_im, mask_im, bvals, bvecs, sigma=None, N=0):
    '''Estimate the tensor image using dipy RESTORE algorithm.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    mask_im: nibabel.Nifti1Image
        A mask NiFTI image 
    bvals : numpy.ndarary
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array. 
    sigma : Optional[numpy.ndarray]
        The noise array. 
    N     : Optional[int]
        Number of coils of the receiver array. Use N = 1 in 
        case of a SENSE reconstruction (Philips scanners) 
        or the number of coils for a GRAPPA reconstruction 
        (Siemens and GE). Use 0 to disable the correction factor, 
        as for example if the noise is Gaussian distributed. 
        
    Returns
    -------
    tuple
        (tensor_im, fa_im, tr_im)  tuple of nibabel.Nifti1Image
    '''
    gtab = gradient_table(bvals.flatten(), bvecs.transpose()) #dipy is transposed
    aff = dwi_im.affine
    hdr = dwi_im.header.copy()
    hdr.set_xyzt_units(hdr.get_xyzt_units()[0],0) # remove seconds from header
    
    dwi_data = dwi_im.get_data()
    mask_data = mask_im.get_data()
    if sigma is None:
        sigma = estimate_noise(dwi_im, N=N)
        # sigma = np.zeros(dwi_im.shape[-1],dtype=float)
        # for idx in range(dwi_im.shape[-1]):
            # sigma[idx] = np.std(dwi_data[...,idx][mask_data > 0])
    
    restore_model = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
    tenfit = restore_model.fit(dwi_data, mask=(mask_im.get_data() > 0)) #saves some time
    
    tensor_data = tenfit.lower_triangular().astype('float32')
    tensor_data = tensor_data[...,np.newaxis,:]
    tensor_data_mask = mask_im.get_data().reshape(list(mask_im.shape) + [1,1])
    tensor_data = tensor_data * tensor_data_mask
    tensor_im = nifti_image(tensor_data, aff, hdr, intent=1005)
    
    fa_data = (tenfit.fa * mask_im.get_data()).astype('float32')
    fa_im = nifti_image(fa_data,aff,hdr, cal_min=0.0, cal_max=1.0)
    
    tr_data = (tenfit.trace * mask_im.get_data()).astype('float32')
    tr_im = nifti_image(tr_data,aff,hdr)
    
    return tensor_im, fa_im, tr_im 
 
class TensorScalarCalculator(object):
    def __init__(self, tensor_im, mask_im=None):
        self.tensor_im = tensor_im
        self._shape = self.tensor_im.shape[:3]
        self._affine = tensor_im.affine
        self._header = tensor_im.header.copy()
        self._header['dim'][3] = 3
        self._header['dim'][5] = 1
        self._header['cal_min'] = 0
        self._header['cal_max'] = 0
        self._header['intent_code'] = 0
        if mask_im:
            self._mask = (mask_im.get_data() > 0)
        else:
            self._mask = (np.sum(np.squeeze(np.abs(self.tensor_im.get_data())) != 0,axis=-1) > 0)
        # self._data = dti.from_lower_triangular(self.tensor_im.get_data()[self._mask,np.newaxis,np.newaxis])
        self._data = dti.from_lower_triangular(self.tensor_im.get_data()[self._mask,0,:])
        self._w, self._v = dti.decompose_tensor(self._data)
        self._w = np.squeeze(self._w)
        self._v = np.squeeze(self._v)
        
    @property
    def MD(self):
        ''' Mean diffusivity. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.mean_diffusivity(self._w,axis=-1)
        return nifti_image(ret, self._affine, self._header)

    @property
    def FA(self):
        ''' Fractional Anisotropy. '''
        ret = np.zeros(self._shape, dtype=np.float32)
        ret[self._mask] = dti.fractional_anisotropy(self._w, axis=-1)
        hdr = self._header.copy()
        hdr['cal_max'] = 1 
        return nifti_image(ret, self._affine, hdr)

    @property
    def TR(self):
        ''' Trace. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.trace(self._w,axis=-1)
        return nifti_image(ret, self._affine, self._header)

    @property
    def AX(self):
        ''' Axial Diffusivity. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.axial_diffusivity(self._w, axis=-1)
        return nifti_image(ret,self._affine, self._header)

    @property
    def RAD(self):
        ''' Radial Diffusivity. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.radial_diffusivity(self._w, axis=-1)
        return nifti_image(ret,self._affine, self._header)

    @property
    def CL(self):
        '''
        Magnetic resonance imaging shows orientation and asymmetry of white matter fiber tracts.
        Peled S1, Gudbjartsson H, Westin CF, Kikinis R, Jolesz FA.
        '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.linearity(self._w, axis=-1)
        return nifti_image(ret,self._affine, self._header)
    
    @property
    def CP(self):
        '''
        Magnetic resonance imaging shows orientation and asymmetry of white matter fiber tracts.
        Peled S1, Gudbjartsson H, Westin CF, Kikinis R, Jolesz FA.
        '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.planarity(self._w, axis=-1)
        return nifti_image(ret,self._affine, self._header)
    
    @property
    def CS(self):
        '''
        Magnetic resonance imaging shows orientation and asymmetry of white matter fiber tracts.
        Peled S1, Gudbjartsson H, Westin CF, Kikinis R, Jolesz FA.
        '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = dti.sphericity(self._w, axis=-1)
        return nifti_image(ret,self._affine, self._header)
    
    @property
    def L1(self):
        ''' Principle eigenvalue. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = self._w[...,0]
        return nifti_image(ret,self._affine, self._header)
    
    @property
    def L2(self):
        ''' 2nd eigenvalue. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = self._w[...,1]
        return nifti_image(ret,self._affine, self._header)
        
    @property
    def L3(self):
        ''' 3rd eigenvalue. '''
        ret = np.zeros(self._shape,dtype=np.float32)
        ret[self._mask] = self._w[...,2]
        return nifti_image(ret,self._affine, self._header)
    
    @property
    def eigenvalues(self):
        return self.L1, self.L2, self.L3
    
    @property
    def V1(self):
        ret = np.zeros(self._shape+(3,),dtype=np.float32)
        ret[self._mask,:] = self._v[...,0,:]
        return nifti_image(ret,self._affine, self._header, intent_code=1007)
    
    @property
    def V2(self):
        ret = np.zeros(self._shape+(3,),dtype=np.float32)
        ret[self._mask,:] = self._v[...,1,:]
        return nifti_image(ret,self._affine, self._header, intent_code=1007)
    
    @property
    def V3(self):
        ret = np.zeros(self._shape+(3,),dtype=np.float32)
        ret[self._mask,:] = self._v[...,2,:]
        return nifti_image(ret,self._affine, self._header, intent_code=1007)
    
    @property
    def eigenvectors(self):
        return self.V1, self.V2, self.V3
    
    @property
    def colormap(self):
        ''' RGB FA-weighted directional colormap. '''
        color_fa = np.zeros(self._shape+(3,),dtype=np.float32)
        color_fa[self._mask, :] = dti.color_fa(dti.fractional_anisotropy(self._w, axis=-1), self._v)
        color_fa = (color_fa * 255).astype('u1')
        rgb_dtype = np.dtype([('R','u1'),('G','u1'),('B','u1')])
        color_fa = np.squeeze(color_fa.view(rgb_dtype))
        hdr = self._header.copy()
        hdr.set_data_dtype(rgb_dtype)
        return nifti_image(color_fa, self._affine, hdr)
 
##############################################
#############    FSL TOOLS    ################
##############################################
 
def bet2_mask_nifti(nifti_im,  f=0.2, g=0.0, erode_iterations=0, return_brain=False):
    '''Mask a nifti image with bet2. Optionally erode the mask by a 2mm sphere
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be masked using fsl bet2
    f : float
        Number to pass to bet2 fractional intensity threshold.
        "fractional intensity threshold (0->1); default=0.5; 
        smaller values give larger brain outline estimates"
    g : float
        Number to pass to bet2 vertical gradient.
        " vertical gradient in fractional intensity threshold (-1->1); default=0; 
        positive values give larger brain outline at bottom, smaller at top"
    erode_iterations : int
        If greater than zero, result of bet2 will be eroded with
        this many iterations of erosion using scipy's binary_erosion
    return_brain : Optional[bool]
        If True, return the brain extracted image in addition to the mask. 
    Returns
    -------
    nibabel.Nifti1Image
        The mask image
    '''
    def erode_mask(nifti_im,connectivity=1,iterations=1):
        from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
        structure = generate_binary_structure(3,connectivity)
        data = (nifti_im.get_data() > 0).astype(np.int16)
        data_eroded = binary_erosion(data, structure, iterations=iterations).astype(np.int16)
        out_im = nifti_image(data_eroded, nifti_im.affine)
        return out_im
        
    tmpdir=make_temp_dir()
    try:
        tmp_filename = os.path.join(tmpdir,'input.nii.gz')
        mask_tmp_betprefix = os.path.join(tmpdir,'input')
        mask_tmp_fileprefix = os.path.join(tmpdir,'input_mask')
        mask_ero_tmp_fileprefix = os.path.join(tmpdir,'input_mask_ero')
        nifti_im.to_filename(tmp_filename)
        cmd = ['bet2',tmp_filename,mask_tmp_betprefix,'-nm','-f',str(float(f)),'-g',str(float(g))]
        ExecCommand(cmd,environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        mask_tmp_filename = mask_tmp_fileprefix+'.nii'
        mask_im = nib.load(mask_tmp_filename)
        if erode_iterations > 0: 
            mask_im = erode_mask(mask_im, iterations = erode_iterations)
        else:
            #make a new image object to load data into memory so tmpdir can be deleted
            #also removes filename attribute of header which refers to a dir which will be removed
            mask_im = nifti_image(mask_im.get_data(), mask_im.affine, mask_im.header)
        if return_brain:
            brain_im = nifti_image(nifti_im.get_data() * mask_im.get_data(), nifti_im.affine, nifti_im.header)
    except Exception as err:
        raise err
    finally:
        shutil.rmtree(tmpdir)
    if return_brain:
        return brain_im, mask_im
    else:
        return mask_im


def decode_phaseenc(x):
    if x.upper() == 'AP':
        return 'j'
    elif x.upper() == 'PA':
        return 'j-'
    elif x.upper() == 'LR':
        return 'i'
    elif x.upper() == 'RL':
        return 'i-'
    elif x.upper() == 'IS':
        return 'k'
    elif x.upper() == 'SI':
        return 'k-'
    else:
        return x
    
def prepare_acqparams_json(json_file, nifti_img, mb_factor=None):
    jdata = json.load(open(json_file))
    phaseenc = jdata['PhaseEncodingDirection']
    eechosp = jdata['EffectiveEchoSpacing']  
    pixdim = nifti_img.shape[:3]
    sign = 1 
    phaseenc = decode_phaseenc(phaseenc)
    try:
        if phaseenc[1] == '-':
            sign = -1
    except:
        pass 
    if phaseenc[0] == 'i':
        matrixpe = pixdim[0]
        triplet = [sign, 0, 0]
    elif phaseenc[0] == 'j':
        matrixpe = pixdim[1]
        triplet = [0, sign, 0]
    elif phaseenc[0] == 'k':
        matrixpe = pixdim[2]
        triplet = [0, 0, sign]
    
    totalreadout = eechosp * (matrixpe-1)
    acqparams_line = triplet + [totalreadout]
    if mb_factor is not None: 
        s = np.array(jdata['SliceTiming'])
        n = len(s)
        indxs = np.arange(n)
        s = s.reshape((int(n/int(mb_factor)),int(mb_factor)), order='F')
        indxs = indxs.reshape((int(n/int(mb_factor)),int(mb_factor)), order='F')
        x = np.concatenate((s,indxs), axis=1)
        x = x[x[:,0].argsort()]
        slspec = x[:,(-1*int(mb_factor)):].astype(int)
        return acqparams_line, slspec 
    else:
        return acqparams_line
    
def prepare_acqparams_nojson(readout_time, phaseenc):   
    phaseenc = decode_phaseenc(phaseenc)
    sign = 1 
    try:
        if phaseenc[1] == '-':
            sign = -1
    except:
        pass 
    if phaseenc[0] == 'i':
        triplet = [sign, 0, 0]
    elif phaseenc[0] == 'j':
        triplet = [0, sign, 0]
    elif phaseenc[0] == 'k':
        triplet = [0, 0, sign]
    acqparams_line = triplet + [readout_time]
    return acqparams_line
    
def prepare_index(bvals):
    # Return index array, which increases by 1 for each encountered b0 image.
    # Weighted images between b0 scans are assigned index corresponding to previous b0 image.
    index=[]
    j=0
    for b in bvals[0,:]:
        if b == 0:
            j+=1 
        if j<1:
            index.append(1)
        else:
            index.append(j)
    return index 
    
def run_topup(dwi_images, bval_arrays, bvec_arrays, output_base, json_files=[], phase_encs=[], 
                readout_time=0.062, mbfactor=None, slspec=None, concatenate=False, config='b02b0.cnf'):
    if not json_files:
        json_files = [None for i in range(len(dwi_images))]
    if not phase_encs:
        phase_encs = [None for i in range(len(dwi_images))]
    dwis_list = []
    acqparams = []
    b0s = [] 
    for dwi, bvals, bvecs, jsonfn, phase_enc in zip(dwi_images, bval_arrays, bvec_arrays, json_files, phase_encs):    
        bvals = round_bvals(bvals)
        dwis_list.append((dwi, bvals, bvecs))
        if jsonfn and (mbfactor is not None):
            acqparams_line, slspec = prepare_acqparams_json(jsonfn, dwi, mbfactor)
        elif jsonfn:
            # use user provided slspec file 
            acqparams_line = prepare_acqparams_json(jsonfn, dwi)
        elif (jsonfn is None):
            acqparams_line = prepare_acqparams_nojson(readout_time, phase_enc)
        # extract b0 
        b0 = extract_b0(dwi, bvals, average=False)
        b0s.append(b0)
        numb0s = 1 if len(b0.shape)==3 else b0.shape[-1]
        for i in range(numb0s):
            acqparams.append(acqparams_line)
    # concatenate b0s and dwis
    b0_img = concatenate_niftis(*b0s)
    if concatenate:
        dwi, bvals, bvecs = concatenate_dwis(*dwis_list)
        write_dwi(output_base+'_DWI_concatenated.nii.gz', dwi, bvals, bvecs)    
    else:
        dwi, bvals, bvecs = dwis_list[0]
    index = prepare_index(bvals)
    # save b0 image, acqparams, index, slspec to txt 
    b0_img.to_filename(output_base+'_b0s.nii.gz')
    np.savetxt(output_base+'_acqparams.txt', np.array(acqparams), delimiter=' ', fmt='%i %i %i %0.8f')
    np.savetxt(output_base+'_index.txt', np.array(index).reshape((1,len(index))), delimiter=' ', fmt='%i')
    if slspec is not None:
        np.savetxt(output_base+'_slspec.txt', np.array(slspec), delimiter=' ', fmt='%i')
     
    # prepare topup command 
    topup_cmd = ['topup', '--imain={}_b0s.nii.gz'.format(output_base),
                '--datain={}_acqparams.txt'.format(output_base),
                '--config={}'.format(config),
                '--out={}'.format(output_base),
                '--fout={}_field'.format(output_base),
                '--iout={}_b0u'.format(output_base),
                '--verbose'
    ]
    with open(output_base+'_command.txt', 'w') as fid:
        fid.write((' '.join(topup_cmd))+'\n')
    
    # run topup 
    logging.info("Run topup")
    ExecCommand(topup_cmd, environ={'FSLOUTPUTTYPE':'NIFTI_GZ'}).run()  
    
def fsl_eddy_correct(dwi_im, bvals):
    '''Runs FSL eddy_correct on a DWI image.
    
    Parameters
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI image
    bvals : numpy.ndarray
        The 1xN bvals array
        
    Returns
    -------
    nibabel.Nifti1Image
        The output DWI image. Bvals remain the same. 
    '''
    tmpdir = make_temp_dir(prefix='fsl_eddy_correct')
    try:
        dwi_tmp_filename=os.path.join(tmpdir,'dwi.nii')
        eddy_out_filebase=os.path.join(tmpdir,'output')
        eddy_out_filename=os.path.join(tmpdir,'output.nii')
        dwi_im.to_filename(dwi_tmp_filename)
        bval_index = np.where(bvals == 0)[1][0] #first b0 image as reference. 
        cmd=['eddy_correct',dwi_tmp_filename,eddy_out_filebase,str(bval_index)]
        returncode, stdout, stderr = ExecCommand(cmd,environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        if returncode != 0:
            sys.stderr.write(stderr)
            raise DiciphrException('Error running eddy_correct cmd {}'.format(cmd))
        dwi_eddy_im = nib.load(eddy_out_filename)
        out_im = nifti_image(dwi_eddy_im.get_data(),dwi_eddy_im.affine,dwi_eddy_im.header)
    except Exception as err:
        raise err
    finally:
        shutil.rmtree(tmpdir)
    return out_im
    
def fsl_eddy_rotate_bvecs(bvecs, eddy_params):
    # Code from scilpy by Maxime Descouteaux 
    logging.debug('diciphr.diffusion.fsl_eddy_rotate_bvecs')
    eddy_a = np.array(eddy_params)
    bvecs = bvecs.transpose()
    bvecs_rotated = np.zeros(bvecs.shape)
    norm_diff = np.zeros(bvecs.shape[0])
    angle = np.zeros(bvecs.shape[0])
    # Documentation here: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq
    # Will eddy rotate my bvecs for me?
    # No, it will not. There is nothing to prevent you from doing that
    # yourself though. eddy produces two output files, one with the corrected
    # images and another a text file with one row of parameters for each volume
    # in the --imain file. Columns 4-6 of these rows pertain to rotation
    # (in radians) around around the x-, y- and z-axes respectively.
    # eddy uses "pre-multiplication".
    # IMPORTANT: from various emails with FSL's people, we couldn't directly
    #            get the information about the handedness of the system.
    #            From the FAQ linked earlier, we deduced that the system
    #            is considered left-handed, and therefore the following
    #            matrices are correct.
    logging.info('Rotating bvecs with the eddy parameters')
    for i in range(len(bvecs)):
        theta_x = eddy_a[i, 3]
        theta_y = eddy_a[i, 4]
        theta_z = eddy_a[i, 5]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), np.sin(theta_x)],
                       [0, -np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                       [0, 1, 0],
                       [np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z), np.sin(theta_z), 0],
                       [-np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])
        v = bvecs[i, :]
        RotationMatrix = np.linalg.inv(np.dot(np.dot(Rx, Ry), Rz))
        v_rotated = np.dot(RotationMatrix, v)
        bvecs_rotated[i, :] = v_rotated
        norm_diff[i] = np.linalg.norm(v - v_rotated)
        if np.linalg.norm(v):
            angle[i] = np.arctan2(np.linalg.norm(np.cross(v, v_rotated)), np.dot(v, v_rotated))
    logging.info('{0} mm is the maximum translation error'.format(np.max(norm_diff)))
    logging.info('{0} degrees is the maximum rotation error'.format(np.max(angle)*180/np.pi))
    return bvecs_rotated.transpose()    
      
def fsl_eddy_post_topup(dwi_im, bvals, bvecs, topup_prefix, acqparamstxt, indextxt, mask_im, replace_outliers=True, workdir=None):
    try:
        eddy_exe = which('eddy_openmp')
    except DiciphrException:
        eddy_exe = which('eddy')
    if workdir is None:
        tmpdir = make_temp_dir(prefix='fsl_topup_eddy')
    else:
        if os.path.isdir(workdir) and os.access(workdir,os.W_OK):
            logging.info("Using working directory {} will not delete.".format(workdir))
            tmpdir = workdir
        else:
            raise DiciphrException("Working directory {} does not exist or is not writable".format(workdir))            
    try:
        dwi_filename = os.path.join(tmpdir, 'dwi.nii.gz')
        bvals_filename = os.path.join(tmpdir, 'dwi.bval')
        bvecs_filename = os.path.join(tmpdir, 'dwi.bvec')
        mask_filename = os.path.join(tmpdir, 'mask.nii.gz')
        eddy_prefix = os.path.join(tmpdir, 'eddy_results')
        
        # X-flip bvec, if data is LPS
        orient_code = ''.join(nib.aff2axcodes(dwi_im.affine))
        if orient_code == 'LPS':
            bvecs = bvecs*np.array([-1,1,1])[...,None] 
        bvals = round_bvals(bvals)
        write_dwi(dwi_filename, dwi_im, bvals, bvecs)
        write_nifti(mask_filename, mask_im)
        
        eddy_cmd=[eddy_exe,
                    '--imain={}'.format(dwi_filename),
                    '--mask={}'.format(mask_filename),
                    '--acqp={}'.format(acqparamstxt),
                    '--index={}'.format(indextxt),
                    '--bvecs={}'.format(bvecs_filename),
                    '--bvals={}'.format(bvals_filename),
                    '--out={}'.format(eddy_prefix),
                    '--topup={}'.format(topup_prefix),
                    '--data_is_shelled',
                    '--very_verbose', 
                ]
        if replace_outliers: 
            eddy_cmd.append('--repol')
        
        ExecCommand(eddy_cmd, environ={'FSLOUTPUTTYPE':'NIFTI_GZ'}).run()
        returncode, stdout, stderr = ExecCommand(eddy_cmd, environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        eddy_params = np.loadtxt('{}.eddy_parameters'.format(eddy_prefix))
        eddy_movement_rms = np.loadtxt('{}.eddy_corrected_data.eddy_movement_rms'.format(eddy_prefix))
        eddy_restricted_movement_rms = np.loadtxt('{}.eddy_corrected_data.eddy_restricted_movement_rms'.format(eddy_prefix))
        # eddy_bvecs = np.loadtxt('eddy_corrected_data.rotated_bvecs')
        # outlier_report = np.loadtxt('eddy_corrected_data.eddy_outlier_report')
        # flip X back for  LPS data to eddy 
        # eddy_bvecs = eddy_bvecs*np.array([-1,1,1])[...,None] 
        dwi_eddy_im = nib.load('{}.nii.gz'.format(eddy_prefix))
        hdr = dwi_eddy_im.header
        affine = dwi_eddy_im.affine
        hdr.set_sform(affine)
        hdr.set_qform(affine)
        out_im = nifti_image(dwi_eddy_im.get_data(), affine, hdr)
        bvecs = fsl_eddy_rotate_bvecs(bvecs, eddy_params)
        if orient_code == 'LPS':
            bvecs = bvecs*np.array([-1,1,1])[...,None] 
        
    except Exception as err:
        raise err
    finally:
        if workdir is None:
            shutil.rmtree(tmpdir)
    return dwi_eddy_im, bvals, bvecs, eddy_params, eddy_movement_rms, eddy_restricted_movement_rms
    
def fsl_eddy(dwi_im, bvals, bvecs, mask_im, readout_time=0.062, replace_outliers=False, workdir=None):
    logging.debug('diciphr.diffusion.fsl_eddy')
    try:
        eddy_exe = which('eddy_openmp')
    except DiciphrException:
        eddy_exe = which('eddy')
    if workdir is None:
        tmpdir = make_temp_dir(prefix='fsl_eddy')
    else:
        if os.path.isdir(workdir) and os.access(workdir,os.W_OK):
            logging.info("Using working directory {} will not delete.".format(workdir))
            tmpdir = workdir
        else:
            raise DiciphrException("Working directory {} does not exist or is not writable".format(workdir))            
    origDir = os.getcwd()
    try:
        orient_code = ''.join(nib.aff2axcodes(dwi_im.affine))
        if orient_code == 'LPS':
            # flip X for LPS data to eddy 
            bvecs = bvecs*np.array([-1,1,1])[...,None] 
        write_dwi(os.path.join(tmpdir, 'dwi.nii'), dwi_im, bvals, bvecs)
        mask_hdr = nib.Nifti1Header()
        mask_hdr.set_sform(mask_im.affine)
        mask_hdr.set_qform(mask_im.affine)
        mask_hdr.set_data_dtype(np.int16)
        mask_im = nifti_image((mask_im.get_data() > 0).astype(np.int16), mask_im.affine, mask_hdr)
        mask_im.update_header()
        write_nifti(os.path.join(tmpdir, 'mask.nii'), mask_im)
        os.chdir(tmpdir)
        indx_text = 'index.txt'
        acqp_text = 'acqparams.txt'
        nb_imgs = dwi_im.shape[-1]
        indx_list = [1 for i in range(nb_imgs)]
        with open(indx_text, 'w') as fid:
            fid.write(' '.join(map(str,indx_list)))
        with open(acqp_text, 'w') as fid:
            fid.write(' '.join(map(str,[0, 1, 0, readout_time])))
        logging.debug('Contents of tmpdir: {}'.format(os.listdir('.')))
        cmd=[eddy_exe,
                '--imain=dwi.nii',
                '--mask=mask.nii',
                '--acqp=acqparams.txt',
                '--index=index.txt',
                '--bvecs=dwi.bvec',
                '--bvals=dwi.bval',
                '--out=eddy_corrected_data',
                '--very_verbose',
                '--data_is_shelled',
            ]
        if replace_outliers:
            cmd.extend(['--repol'])
        returncode, stdout, stderr = ExecCommand(cmd, environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        eddy_params = np.loadtxt('eddy_corrected_data.eddy_parameters')
        eddy_movement_rms = np.loadtxt('eddy_corrected_data.eddy_movement_rms')
        eddy_restricted_movement_rms = np.loadtxt('eddy_corrected_data.eddy_restricted_movement_rms')
        # eddy_bvecs = np.loadtxt('eddy_corrected_data.rotated_bvecs')
        # outlier_report = np.loadtxt('eddy_corrected_data.eddy_outlier_report')
        # flip X back for  LPS data to eddy 
        # eddy_bvecs = eddy_bvecs*np.array([-1,1,1])[...,None] 
        dwi_eddy_im = nib.load('eddy_corrected_data.nii')
        hdr = dwi_eddy_im.header
        affine = dwi_eddy_im.affine
        hdr.set_sform(affine)
        hdr.set_qform(affine)
        out_im = nifti_image(dwi_eddy_im.get_data(), affine, hdr)
        bvecs = fsl_eddy_rotate_bvecs(bvecs, eddy_params)
        if orient_code == 'LPS':
            bvecs = bvecs*np.array([-1,1,1])[...,None] 
    finally:
        os.chdir(origDir)
        if workdir is None:
            shutil.rmtree(tmpdir)
    return out_im, bvals, bvecs, eddy_params, eddy_movement_rms, eddy_restricted_movement_rms

###########################
### Mrtrix/tractography ###
###########################
    
def threshold_fa(fa_im, threshold=0.7, brain_mask_im=None, erode_iterations=1):
    '''Generate a mask of high-FA voxels. Typically used for FOD response function calculation. 
        
    Parameters
    ----------
    fa_im : nibabel.Nifti1Image
        The FA image
    threshold : Optional[float]
        The FA value at which to threshold voxels. Default 0.7 
    brain_mask_im : Optional[numpy.ndarray]
        The brain mask image
    erode_iterations : Optional[int]
        Number of times to erode the brain mask, used to locate periperhal voxels
        
    Returns
    -------
    nibabel.Nifti1Image
        The single fiber mask.
    '''
    if brain_mask_im is None:
        brain_mask_im = nifti_image((fa_im.get_data() > 0).astype(np.int16), fa_im.affine)
    if erode_iterations > 0:
        brain_mask_ero_im = erode_image(brain_mask_im, iterations=erode_iterations)
    fa_data = fa_im.get_data()
    brain_mask_ero_data = brain_mask_ero_im.get_data()
    single_fiber_data = (brain_mask_ero_data*(fa_data > threshold)).astype(np.int16)
    single_fiber_im = nifti_image(single_fiber_data, fa_im.affine)
    return single_fiber_im
    
def angle_between(vector1, vector2): 
    '''Returns the angle in radians between vectors 'vector1' and 'vector2'.'''
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def calculate_lmax_from_bvecs(bvecs):
    '''Given a dwi_im, calculate the largest value L for constrained spherical deconvolution.'''
    return calculate_lmax(number_unique_gradients(bvecs))

def calculate_lmax(num_obs):
    '''Given a number of observations, calculate the largest value L for constrained spherical deconvolution.'''
    L = 2
    min_num_obs = (L+1)*(L+2)/2
    if min_num_obs > num_obs: 
        raise DiciphrException('Insufficient number dwi directions for smallest value L of 2') 
    while True:
        next_L = L+2
        next_min_num_obs = (next_L+1)*(next_L+2)/2
        if next_min_num_obs <= num_obs:
            L = next_L 
            min_num_obs = next_min_num_obs
        else:
            break
    return L 
    
####################################
#######    MISCELLANEOUS    ########
####################################

def dwi_multi_b0_temporal_snr(dwi_im, bvals, bvecs, mask_im):
    tol=20
    b0_indices = np.where(bvals.flatten() < tol)
    if len(b0_indices[0]) < 3:
        raise DiciphrException('Cannot calculate temporal SNR with too few b-zero images.')
    data=dwi_im.get_data()
    affine=dwi_im.affine
    mask_data=mask_im.get_data()
    b0s=data[:,:,:,b0_indices]
    std=np.std(b0s, axis=-1)
    mn=np.mean(b0s, axis=-1)
    tsnr=mn/std
    tsnr[np.logical_not(mask_data)] = 0
    tsnr_img = nifti_image(tsnr, affine)
    return tsnr_img
    
def dti_roi_stats(tensor_im, atlas_im, labels=None, scalars=['FA','TR','AX','RAD'], measures=['mean','median','std'], nonzero=True, min_roi_size=20, mask_im=None):
    logging.debug('diciphr.diffusion.dti_roi_stats')
    from collections import OrderedDict
    from itertools import product
    tensor_affine = tensor_im.affine
    atlas_affine = atlas_im.affine
    tensor_shape = tensor_im.shape
    atlas_shape = atlas_im.shape
    
    if np.sum(np.abs(tensor_affine - atlas_affine)) > 1e-6 or tensor_shape[:3] != atlas_shape:
        raise DiciphrException("Tensor image and atlas are not in the same space.") 
    if not is_tensor(tensor_im):
        raise DiciphrException("Input Nifti image is not tensor.") 
    for _sc in scalars:
        if not _sc in ['FA','TR','AX','RAD','MD','CL','CP','CS']:
            raise DiciphrException("Requested an unrecognized scalar code : {}.".format(_sc))
    for _m in measures:
        if not _m in ['mean','median','std']:
            raise DiciphrException("Requested an unrecognized measure : {}.".format(_m))
            
    C = TensorScalarCalculator(tensor_im, mask_im=mask_im)
    atlas_data = atlas_im.get_data()
    if not labels:
        labels = [ int(_lbl) for _lbl in np.unique(atlas_data[atlas_data>0]) ]
    else:
        labels = list(map(int,labels))
    #make tuple-keyed dictionary as return
    ret = {} 
    for _sc in scalars:
        _scalar_im = C.__getattribute__(_sc)
        _scalar_dict = scalar_roi_stats(_scalar_im, atlas_im, labels=labels, measures=measures, nonzero=nonzero, min_roi_size=min_roi_size, mask_im=mask_im)
        ret[(_m,_sc)] = _scalar_dict[_m]
    return ret
        
def scalar_roi_stats(scalar_im, atlas_im, labels=None, measures=['mean','median','std'], nonzero=True, min_roi_size=20, mask_im=None):
    from collections import OrderedDict
    
    ### TO DO : args *scalar_ims 
    scalar_data = scalar_im.get_data()
    atlas_data = atlas_im.get_data().astype(int)
    if mask_im is not None:
        mask = mask_im.get_data() > 0 
    else:
        mask = atlas_data > 0
    if nonzero:
        mask = np.logical_and(mask, scalar_data != 0)
    if labels is None:
        labels = range(1,atlas_data.max()+1)
    num_labels = len(labels)
    atlas_volumes=[]
    scalar_data = scalar_im.get_data().astype(np.float32)[mask]
    scalar_means=np.zeros(num_labels,)
    scalar_stds=np.zeros(num_labels,)
    scalar_medians=np.zeros(num_labels,)
    atlas_data = atlas_data[mask]
    
    for i,lbl in enumerate(labels):
        atlas_volume = np.sum(atlas_data == lbl)
        atlas_volumes.append(atlas_volume)
        if atlas_volume < min_roi_size:
            atlas_data[atlas_data == lbl] = 0 
            continue
        try:
            scalar_means[i] = np.mean(scalar_data[atlas_data==lbl])
            scalar_medians[i] = np.median(scalar_data[atlas_data==lbl])
            scalar_stds[i] = np.std(scalar_data[atlas_data==lbl], ddof=1)
        except:
            continue
    scalar_means[np.isnan(scalar_means)]=0
    scalar_medians[np.isnan(scalar_medians)]=0
    scalar_stds[np.isnan(scalar_stds)]=0    
    ret = {}
    if 'mean' in measures: 
        ret['mean'] = OrderedDict(zip(labels,scalar_means))
    if 'median' in measures:
        ret['median'] = OrderedDict(zip(labels,scalar_medians))
    if 'std' in measures:
        ret['std'] = OrderedDict(zip(labels,scalar_stds))
    return ret 

def calc_motion(eddy_params_filename):
    motion_table = np.loadtxt(filename)[:,:6]
    rel_ = [] 
    abs_ = [] 
    xa0, ya0, za0, aa0, ba0, ca0 = motion_table[0,:]
    for row_index in range(1,motion_table.shape[0]):
        x0, y0, z0, a0, b0, c0 = motion_table[row_index - 1,:]
        x1, y1, z1, a1, b1, c1 = motion_table[row_index,:]
        # radian angles -> 50 mm circle displacements
        d0 = np.abs(x0 - x1)
        d1 = np.abs(y0 - y1)
        d2 = np.abs(z0 - z1)
        d3 = np.abs(a0 - a1) * 50.0
        d4 = np.abs(b0 - b1) * 50.0
        d5 = np.abs(c0 - c1) * 50.0
        fdr = d0 + d1 + d2 + d3 + d4 + d5 
        rel_.append(fdr)
      
        da0 = np.abs(xa0 - x1)
        da1 = np.abs(ya0 - y1)
        da2 = np.abs(za0 - z1)
        da3 = np.abs(aa0 - a1) * 50.0
        da4 = np.abs(ba0 - b1) * 50.0
        da5 = np.abs(ca0 - c1) * 50.0
        fda = da0 + da1 + da2 + da3 + da4 + da5 
        abs_.append(fda)
  
    ## Convert to array to calculate rms
    abs_array=np.array(abs_)
    rel_array=np.array(rel_)

    ## Calculating rms displacement
    abs_rms=np.sqrt(np.mean(abs_array**2))
    rel_rms=np.sqrt(np.mean(rel_array**2))
    return abs_rms, rel_rms 
