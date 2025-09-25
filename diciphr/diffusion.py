# -*- coding: utf-8 -*-
import os, shutil, logging
import numpy as np
import nibabel as nib
from dipy.reconst import dti
from dipy.core.gradients import gradient_table
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise import noise_estimate
from dipy.denoise.localpca import localpca, mppca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate 
from dipy.denoise.gibbs import gibbs_removal
from diciphr.utils import ( make_dir, force_to_list, read_json_file, 
                TempDirManager, ExecCommand, ExecFSLCommand, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, write_nifti, nifti_image, 
                erode_image, check_affines_and_shapes_match, is_valid_dwi, 
                read_dwi, write_dwi, bet2_mask_nifti, mask_image, 
                threshold_image, crop_pad_image, resample_image, 
                smooth_image, split_image, concatenate_niftis )

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
    bvals_out = np.concatenate(bvals_list, axis=1)
    bvecs_out = np.concatenate(bvecs_list, axis=1)
    
    affines=[a.affine for a in dwi_im_list]
    affines_match = True
    for idx in range(len(affines)-1):
        affines_match = affines_match and _affines_match(affines[idx], affines[idx+1])
    if not affines_match:
        logging.warning('Attempting to concatenate nifti files whose affine transformations do not match')
    dwi_im_out = concatenate_niftis(*dwi_im_list)
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
        dat = im.get_fdata()
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
    dwi_data = dwi_im.get_fdata()
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
    data=dwi_im.get_fdata()
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
    
def extract_b0(dwi_im, bvals, bvecs=None, first=False, average=True, mcflirt=False):
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
        If True, return the first encountered B0. 
    average: Optional[bool]
        If True ( default ) averages the B0s, else returns a 4D volume of B0 images.
    mcflirt: Optional[bool]
        If True, run mcflirt before averaging the B0s, default False. 
    Returns
    -------
    nibabel.Nifti1Image
        The average B0 image.
    '''
    logging.debug('diciphr.diffusion.extract_b0')
    data = dwi_im.get_fdata()
    if len(data.shape)==3:
        data = data[...,np.newaxis]
    if first:
        indy = np.where(bvals ==0)[1][0]
        b0 = data[...,indy]
    else:
        indy = np.where(bvals == 0)[1]
        if average:
            if mcflirt:
                b0_img = nifti_image(data[...,indy], dwi_im.affine, dwi_im.header)
                mcf_img = fsl_mcflirt(b0_img)
                b0 = np.mean(mcf_img.get_fdata(),axis=-1)
            else:
                b0 = np.mean(data[...,indy],axis=-1)
        else:
            b0 = data[...,indy]
    b0_im = nifti_image(b0, dwi_im.affine, dwi_im.header)
    return b0_im

def extract_gaussian_shells(dwi_im, bvals, bvecs):
    if np.max(bvals) > 1500 or np.min(bvals[bvals>0]) < 500:
        logging.info("B-values outside the Gaussian range 500-1500 detected.")
        bvals_gaussian = list(np.unique(bvals[np.logical_and(bvals >= 500, bvals <=1500)]))
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
    dwi_data=dwi_im.get_fdata()
    mask_data = mask_im.get_fdata().astype(int)
    mask_data = mask_data[...,np.newaxis]
    dwi_masked_data = dwi_data * mask_data
    dwi_masked_im = nifti_image(dwi_masked_data.astype(dwi_im.get_data_dtype()), dwi_im.affine, dwi_im.header)
    return dwi_masked_im

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
    b0_data = b0_im.get_fdata()
    if wm_im is None:
        fa_wm_threshold = 0.30
        # threshold FA to get WM mask 
        if mask_im is None:
            mask_im = bet2_mask_nifti(b0_im, erode_iterations=1)
        tensor_im, fa_im = estimate_tensor(dwi_im, mask_im, bvals, bvecs, return_fa=True)
        wm_mask = fa_im.get_fdata() > fa_wm_threshold
    else:
        wm_mask = wm_im.get_fdata() > 0 
    b0_wm_median_value = np.median(b0_data[wm_mask])
    scale = reference_value / b0_wm_median_value
    dwi_data_scaled = dwi_im.get_fdata() * scale
    dwi_out_im = nifti_image(dwi_data_scaled, affine)
    return dwi_out_im, bvals, bvecs 

def compute_suggested_patch_radius_3d(arr):
    """
    Compute the suggested patch radius.
    """
    root = np.ceil(arr.shape[-1] ** (1.0 / 3))  # 3D
    root = root + 1 if (root % 2) == 0 else root  # make odd
    return int((root - 1) / 2)

def mppca_denoise(dwi_im, bvals, bvecs, return_diff=False, patch_radius=2):
    logging.debug('diciphr.diffusion.mppca_denoise')
    data = dwi_im.get_fdata()
    hdr = dwi_im.header 
    affine = dwi_im.affine
    suggested_patch_radius = compute_suggested_patch_radius_3d(data)
    if suggested_patch_radius > patch_radius:
        patch_radius = suggested_patch_radius
    denoised_arr = mppca(data, patch_radius=patch_radius)
    dwi_denoised_im = nifti_image(denoised_arr, affine, hdr)
    if return_diff:
        diff_im = nifti_image(dwi_im.get_fdata() - dwi_denoised_im.get_fdata(), affine, hdr)
        return dwi_denoised_im, diff_im
    else:
        return dwi_denoised_im

def lpca_denoise(dwi_im, bvals, bvecs, return_diff=False, smooth=3, tau_factor=2.3, patch_radius=2):
    logging.debug('diciphr.diffusion.lpca_denoise')
    data = dwi_im.get_fdata()
    hdr = dwi_im.header 
    affine = dwi_im.affine
    gtab = gradient_table(bvals.flatten(), bvecs.T)
    logging.info('Estimate sigma for pca')
    sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=smooth)
    logging.info('Local PCA denoising')
    suggested_patch_radius = compute_suggested_patch_radius_3d(data)
    if suggested_patch_radius > patch_radius:
        patch_radius = suggested_patch_radius
    denoised_arr = localpca(data, sigma, tau_factor=tau_factor, patch_radius=patch_radius)
    logging.info('Local PCA done.')
    dwi_denoised_im = nifti_image(denoised_arr, affine, hdr)
    if return_diff:
        diff_im = nifti_image(dwi_im.get_fdata() - dwi_denoised_im.get_fdata(), affine, hdr)
        return dwi_denoised_im, diff_im
    else:
        return dwi_denoised_im
        
def gibbs_unringing(dwi_img, acquisition_slicetype='axial', n_points=3, num_processes=1, return_diff=False):
    data = dwi_img.get_fdata()
    logging.info('Begin denoising algorithm')
    if acquisition_slicetype.lower()[0] == 'a':
        slice_axis=2
    elif acquisition_slicetype.lower()[0] == 'c':
        slice_axis=1
    elif acquisition_slicetype.lower()[0] == 's':
        slice_axis=0
    else:
        raise ValueError('Unrecognized acquisition slice type: {}'.format(acquisition_slicetype))
    corrected = gibbs_removal(data, inplace=False, 
        slice_axis=slice_axis, n_points=n_points, num_processes=num_processes)
    result_img = nifti_image(corrected, dwi_img.affine)
    if return_diff:
        difference = data - corrected 
        diff_img = nifti_image(difference, dwi_img.affine)
        return result_img, diff_img
    else:
        return result_img 
    
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
            sigma[...,idx] = noise_estimate.piesno(dwi_im.get_fdata()[...,idx,:], N=N, return_mask=False)
        logging.debug("Noise standard deviation from piesno is {}".format(sigma[0,0,:]))
    else:
        sigma = noise_estimate.estimate_sigma(dwi_im.get_fdata(), N=N)
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
        mask=mask_im.get_fdata()
    dwi_data = dwi_im.get_fdata()
    logging.debug("Running nlmeans with rician {}".format(rician))
    dwi_denoised_data = nlmeans(dwi_data, sigma, mask=mask, rician=rician)
    dwi_denoised_data = dwi_denoised_data.astype(dwi_data.dtype)
    dwi_denoised_im = nifti_image(dwi_denoised_data,dwi_im.affine)
    return dwi_denoised_im
    
def adjust_b0_problem_voxels(dwi_im, bvals, lmin=1e-9):
    # Adjust B0 voxels that are too low based on tissue properties 
    # i.e. such that the least signal attenuation has a diffusivity of lmin
    dwi = dwi_im.get_fdata()
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
   
def estimate_tensor(dwi_im, mask_im, bvals, bvecs, fit_method='WLS', return_fa=False):
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
    return_fa : Optional[bool]
        If true, return the FA 
        
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
    tenfit = tenmodel.fit(dwi_im.get_fdata(), mask=(mask_im.get_fdata() > 0)) #saves some time
    
    tensor_data = tenfit.lower_triangular().astype('float32')
    tensor_data = tensor_data[...,np.newaxis,:]
    tensor_data_mask = mask_im.get_fdata().reshape(list(mask_im.shape) + [1,1])
    tensor_data = tensor_data * tensor_data_mask
    tensor_im = nifti_image(tensor_data, aff, hdr, intent=1005)
    
    if return_fa:
        fa_data = (tenfit.fa * mask_im.get_fdata()).astype('float32')
        fa_im = nifti_image(fa_data,aff,hdr, cal_min=0.0, cal_max=1.0)
        return tensor_im, fa_im
    else:
        return tensor_im
    
def estimate_tensor_restore(dwi_im, mask_im, bvals, bvecs, sigma=None, N=0, return_fa=False):
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
    return_fa : Optional[bool]
        If true, return the FA 
        
    Returns
    -------
    tuple
        (tensor_im, fa_im, tr_im)  tuple of nibabel.Nifti1Image
    '''
    gtab = gradient_table(bvals.flatten(), bvecs.transpose()) #dipy is transposed
    aff = dwi_im.affine
    hdr = dwi_im.header.copy()
    hdr.set_xyzt_units(hdr.get_xyzt_units()[0],0) # remove seconds from header
    
    dwi_data = dwi_im.get_fdata()
    mask_data = mask_im.get_fdata()
    if sigma is None:
        sigma = estimate_noise(dwi_im, N=N)
        # sigma = np.zeros(dwi_im.shape[-1],dtype=float)
        # for idx in range(dwi_im.shape[-1]):
            # sigma[idx] = np.std(dwi_data[...,idx][mask_data > 0])
    
    restore_model = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
    tenfit = restore_model.fit(dwi_data, mask=(mask_data > 0)) #saves some time
    
    tensor_data = tenfit.lower_triangular().astype('float32')
    tensor_data = tensor_data[...,np.newaxis,:]
    tensor_data_mask = mask_data.reshape(list(mask_im.shape) + [1,1])
    tensor_data = tensor_data * tensor_data_mask
    tensor_im = nifti_image(tensor_data, aff, hdr, intent=1005)
    if return_fa:
        fa_data = (tenfit.fa * mask_data).astype('float32')
        fa_im = nifti_image(fa_data,aff,hdr, cal_min=0.0, cal_max=1.0)
        return tensor_im, fa_im
    else:
        return tensor_im
        
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
            self._mask = (mask_im.get_fdata() > 0)
        else:
            self._mask = (np.sum(np.squeeze(np.abs(self.tensor_im.get_fdata())) != 0,axis=-1) > 0)
        self._data = dti.from_lower_triangular(self.tensor_im.get_fdata()[self._mask,0,:])
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
def decode_phaseenc(x):
    # DICIPHR PIPELINE: Our data are in LPS orientation 
    # https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=FSL;f63aa333.1701
    #You have specified that the phase-encoding is in the y-direction, and that is correct. As for the signs it doesn’t really matter if you specify
    #0 -1 0 t
    #0 1 0 t
    #or 
    #0 1 0 t
    #0 -1 0 t
    #as long as you are consistent between topup and applytopup (or eddy). 
    #Of course one of them is “correct” in that the field has the correct sign, 
    #and the easiest way to check that is to look at your output field. 
    #Re-run topup with --fout=my_field and take a look at my_field.nii.gz. 
    #If you see hotspots (high field) along the ear canals and around the sinuses, then it is correct. 
    #If you see coldspots (low field) in those places, then it is sign reversed.
    #Jesper
    # checked LR and RL (PPMI) - field was bright in sinus and dark in central sulcus 
    # checked AP and PA (Pilot) - field was bright in sinus and dark in central sulcus 
    mapping = {
        'AP': 'j-',
        'PA': 'j',
        'LR': 'i',
        'RL': 'i-',
        'IS': 'k',
        'SI': 'k-',
    }
    return mapping.get(x.strip().upper(),x) # return original if not found 

def group_diffusion_niftis_json(nifti_files, json_files, bval_files=[None], bvec_files=[None]):
    grouped = {}
    for nifti, js, bval, bvec in zip(nifti_files, json_files, bval_files, bvec_files):
        jdata = read_json_file(js)
        ped = jdata.get('PhaseEncodingDirection')
        trt = jdata.get('TotalReadoutTime')
        key = (ped, trt)
        if key not in grouped.keys():
            grouped[key] = []
        grouped[key].append((nifti, js, bval, bvec))
    return grouped 
    
def prepare_acqparams_json(json_file, nifti_img, mb_factor=None):
    # read json file 
    jdata = read_json_file(json_file)
    # get relevant fields 
    phaseenc = jdata.get('PhaseEncodingDirection')
    totalreadout = jdata.get('TotalReadoutTime')
    eechosp = jdata.get('EffectiveEchoSpacing')
    pe_steps = jdata.get('PhaseEncodingSteps')
    pixdim = nifti_img.shape[:3] 
    if phaseenc is None:
        raise ValueError(f"No PhaseEncodingDirection field in file {json_file}")    
    # decode phase enc and set up look-up tables 
    phaseenc = decode_phaseenc(phaseenc)
    
    sign = 1 
    if len(phaseenc) > 1 and phaseenc[1] == '-':
        sign = -1    
    matrix_pe_dict = {'i': pixdim[0],
                      'j': pixdim[1],
                      'k': pixdim[2]}
    triplets_dict =  {'i': [sign, 0, 0],
                      'j': [0, sign, 0],
                      'k': [0, 0, sign]}
    # build acqparams line from info we have 
    if totalreadout is None:
        if pe_steps is None:
            pe_steps = matrix_pe_dict[phaseenc[0]]
        if eechosp is None:
            raise ValueError(f"Insufficient information to get readout time from file {json_file}")
        totalreadout = float(eechosp) * (float(pe_steps) - 1)
    acqparams_line = triplets_dict[phaseenc[0]]
    acqparams_line.append(totalreadout)
    logging.debug(f"{acqparams_line}")
    if mb_factor is not None: 
        # build slspec array 
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
    if len(phaseenc) > 1 and phaseenc[1] == '-':
        sign = -1    
    triplets_dict =  {'i': [sign, 0, 0],
                      'j': [0, sign, 0],
                      'k': [0, 0, sign]}
    acqparams_line = triplets_dict[phaseenc[0]]
    acqparams_line.append(readout_time)
    return acqparams_line

def prepare_index(*bval_arrays, average_b0s=False):
    """
    Generates an index list for multiple b-value arrays, identifying and grouping b=0 values.

    Parameters:
    ----------
    *bval_arrays : array-like
        One or more arrays containing b-values
    average_b0s : bool, optional (default=False)
        If True, all b=0 values across arrays are grouped under the same index.
        If False, b=0 values are indexed individually based on their occurrence.

    Returns:
    -------
    index : list of int
        A list of indices corresponding to each b-value in the input arrays.
        - If `average_b0s` is True, all b=0 values are assigned the same index.
        - If `average_b0s` is False, b=0 values are indexed by their order of appearance.
    """
    index=[]
    j=0
    k=0
    for bvals in bval_arrays:
        k+=1
        for b in np.asarray(bvals).flatten():
            if b == 0:
                j+=1 
            if j<1 or average_b0s:
                index.append(k)
            else:
                index.append(j)
    return index 

def most_gradients_pe(bval_arrays, all_acqparams):
    logging.debug('diciphr.diffusion.most_gradients_pe')
    # Count non-zero gradients in each bval array
    nonzero_counts = [np.count_nonzero(bval) for bval in bval_arrays]
    # Accumulate non-zero counts per unique acqparams
    acqparam_counts = {}
    for bval_count, ap in zip(nonzero_counts, all_acqparams):
        key = tuple(ap)
        acqparam_counts[key] = acqparam_counts.get(key, 0) + bval_count
    # Find the maximum count
    max_count = max(acqparam_counts.values())
    # Identify all acqparams that tie for the maximum
    max_aps = {ap for ap, count in acqparam_counts.items() if count == max_count}
    # Return True for entries matching any of the max_aps
    logging.debug(f'{[tuple(ap) in max_aps for ap in all_acqparams]}')
    return [tuple(ap) in max_aps for ap in all_acqparams]

def pad_image_for_topup(nifti_img):
    nifti_img.shape 
    x_adjust = [0,0]
    y_adjust = [0,0]
    z_adjust = [0,0]
    adjusted = False
    if nifti_img.shape[0]%2 != 0:
        x_adjust = [0,1]
        adjusted = True
    if nifti_img.shape[1]%2 != 0:
        y_adjust = [0,1]
        adjusted = True
    if nifti_img.shape[2]%2 != 0:
        z_adjust = [0,1]
        adjusted = True
    if adjusted:
        logging.warning("Inputs to topup need to have even voxel dimensions. Image was padded with adjustments x:{0} y:{1} z:{2}".format(
            x_adjust, y_adjust, z_adjust
        ))
    return crop_pad_image(nifti_img, x_adjust, y_adjust, z_adjust), adjusted
    
def synb0_disco(synb0_sif, fslicense, dwi_img, bvals, acqparams, t1_img, t1_mask_img=None, topup=False):
    acqparams2 = list(acqparams[:3]) + [0.0]
    with TempDirManager(prefix='synb0_disco') as manager:
        tmpdir = manager.path()
        input_tmp_datadir = make_dir(os.path.join(tmpdir, 'input'))
        output_tmp_datadir = make_dir(os.path.join(tmpdir, 'output'))
        # extract the first b0 volume 
        bzero_img = extract_b0(dwi_img, bvals, first=True)
        bzero_img.to_filename(os.path.join(input_tmp_datadir, "b0.nii.gz"))
        if t1_mask_img is not None:
            t1_img = mask_image(t1_img, t1_mask_img)
        t1_img.to_filename(os.path.join(input_tmp_datadir, "T1.nii.gz"))
        np.savetxt(os.path.join(input_tmp_datadir, "acqparams.txt"), 
                       np.asarray([acqparams, acqparams2]), 
                       delimiter=' ', fmt='%i %i %i %0.8f')
        cmd = ["apptainer","run","-e",
            "-B","{0}:/INPUTS".format(input_tmp_datadir),
            "-B","{0}:/OUTPUTS".format(output_tmp_datadir),
            "-B","{0}:/extra/freesurfer/license.txt".format(fslicense),
            synb0_sif]
        if not topup: 
            cmd.extend(["--notopup"])
        if t1_mask_img is not None:
            cmd.append("--stripped")
        ExecCommand(cmd).run()
        result_img = read_nifti(os.path.join(output_tmp_datadir, "b0_u.nii.gz"))
        result_img = mask_image(result_img, result_img) # set anything negative to zero 
    return result_img 

def run_topup(dwi_images, bval_arrays, bvec_arrays, acqparams_list, output_base, 
              keep_dwis=[], slspec=None, average_b0s=False, mcflirt=False, first_b0=False, 
              mbfactor=None, concatenate=False, config=None):
    logging.debug("diciphr.diffusion.run_topup")
    index = []
    acqparams = [] 
    b0_images = [] 
    for dwi_img, bval, bvec, acqparams_line in zip(dwi_images, bval_arrays, bvec_arrays, acqparams_list):
        b0_img = extract_b0(dwi_img, bval, first=first_b0, average=average_b0s, mcflirt=mcflirt)
        b0_img, adjusted = pad_image_for_topup(b0_img)
        b0_images.append(b0_img)
        n = 1 if len(b0_img.shape) == 3 else b0_img.shape[3]
        for nn in range(n):
            acqparams.append(acqparams_line)
    index = prepare_index(*[bval for bval,k in zip(bval_arrays, keep_dwis) if k], average_b0s=average_b0s or first_b0)
    b0_img = concatenate_niftis(*b0_images)    
    # save b0 image, acqparams, index, slspec to txt 
    # b0_img.to_filename(output_base+'_b0s.nii.gz')
    # np.savetxt(output_base+'_acqparams.txt', np.array(acqparams), delimiter=' ', fmt='%i %i %i %0.8f')
    # np.savetxt(output_base+'_index.txt', np.array(index).reshape((1,len(index))), delimiter=' ', fmt='%i')
    # if slspec is not None:
    #     np.savetxt(output_base+'_slspec.txt', np.concatenate((np.array(s) for s in slspec)), delimiter=' ', fmt='%i')
    
    with TempDirManager(prefix='topup') as manager:
        tmpdir = manager.path()
        if config is None:
            config = 'b02b0.cnf'
        else:
            shutil.copyfile(config, os.path.join(tmpdir, 'custom.cnf'))
            config = os.path.join(tmpdir, 'custom.cnf')
        tmpbase = os.path.join(tmpdir, os.path.basename(output_base))
        b0_img.to_filename(tmpbase+'_b0s.nii.gz')
        np.savetxt(tmpbase+'_acqparams.txt', np.array(acqparams), delimiter=' ', fmt='%i %i %i %0.8f')
        np.savetxt(tmpbase+'_index.txt', np.array(index).reshape((1,len(index))), delimiter=' ', fmt='%i')
        if slspec is not None:
            np.savetxt(tmpbase+'_slspec.txt', np.concatenate((np.array(s) for s in slspec)), delimiter=' ', fmt='%i')
        # prepare topup command 
        topup_cmd = ['topup', f'--imain={tmpbase}_b0s.nii.gz',
                    f'--datain={tmpbase}_acqparams.txt',
                    f'--config={config}',
                    f'--out={tmpbase}',
                    f'--fout={tmpbase}_field',
                    f'--iout={tmpbase}_b0u',
                    '--verbose'
        ]
        with open(output_base+'_command.txt', 'w') as fid:
            fid.write((' '.join(topup_cmd))+'\n')
        # run topup 
        logging.info("Run topup")
        ExecFSLCommand(topup_cmd).run()  
        shutil.copyfile(tmpbase+'_movpar.txt', output_base+'_movpar.txt')
        shutil.copyfile(tmpbase+'_acqparams.txt', output_base+'_acqparams.txt')
        shutil.copyfile(tmpbase+'_index.txt', output_base+'_index.txt')
        if slspec is not None:
            shutil.copyfile(tmpbase+'_slspec.txt', output_base+'_slspec.txt')
        # undo cropping/padding if necessary 
        if adjusted:
            field_img = resample_image(read_nifti(tmpbase+'_field.nii.gz'), master=dwi_images[0], interp='NearestNeighbor')
            field_img.to_filename(output_base+'_field.nii.gz')
            fieldcoef_img = resample_image(read_nifti(tmpbase+'_fieldcoef.nii.gz'), master=dwi_images[0], interp='NearestNeighbor')
            fieldcoef_img.to_filename(output_base+'_fieldcoef.nii.gz')
            b0u_img = resample_image(read_nifti(tmpbase+'_b0u.nii.gz'), master=dwi_images[0], interp='NearestNeighbor')
            b0u_img.to_filename(output_base+'_b0u.nii.gz')
        else:
            shutil.copyfile(tmpbase+'_field.nii.gz', output_base+'_field.nii.gz')
            shutil.copyfile(tmpbase+'_fieldcoef.nii.gz', output_base+'_fieldcoef.nii.gz')
            shutil.copyfile(tmpbase+'_b0u.nii.gz', output_base+'_b0u.nii.gz')
    return output_base

def run_topup_post_synb0(dwi_img, bvals, bvecs, synb0_img, acqparams_line, output_base, smooth_fwhm=1.15, config=None):
    with TempDirManager(prefix='synb0_topup') as manager:
        tmpdir = manager.path()
        tmpbase = os.path.join(tmpdir, os.path.basename(output_base))
        bvals = round_bvals(bvals)
        ref_img = split_image(dwi_img, dimension='t', index=0) # original image size 
        logging.debug(f'ref_img.shape : {ref_img.shape}')
        dwi_img, adjusted1 = pad_image_for_topup(dwi_img)
        synb0_img, adjusted2 = pad_image_for_topup(synb0_img)
        logging.debug(f'dwi_img.shape : {dwi_img.shape}')
        logging.debug(f'synb0_img.shape : {synb0_img.shape}')
        b0s = [] 
        acqparams = []
        acqparams_line2 = list(acqparams_line[:3]) + [0.0]
        # extract b0 
        b0 = extract_b0(dwi_img, bvals, first=True)
        if smooth_fwhm>0:
            b0 = smooth_image(b0, smooth_fwhm)
        b0s.append(b0)
        numb0s = 1 if len(b0.shape)==3 else b0.shape[-1]
        for i in range(numb0s):
            acqparams.append(acqparams_line)
        b0s.append(synb0_img)
        acqparams.append(acqparams_line2)
        index = prepare_index(bvals)
        np.savetxt(f"{output_base}_acqparams.txt", np.array(acqparams), delimiter=' ', fmt='%i %i %i %0.8f')
        np.savetxt(f"{output_base}_index.txt", np.array(index).reshape((1,len(index))), delimiter=' ', fmt='%i')
        # concatenate b0s and dwis
        b0_img = concatenate_niftis(*b0s)
        b0_img.to_filename(f"{output_base}_b0s.nii.gz")
        # copy inputs to tmpdir 
        shutil.copyfile(f"{output_base}_b0s.nii.gz", f"{tmpbase}_b0s.nii.gz")
        shutil.copyfile(f"{output_base}_acqparams.txt", f"{tmpbase}_acqparams.txt")
        topup_cmd = ['topup', 
                    f"--imain={tmpbase}_b0s.nii.gz",
                    f"--datain={tmpbase}_acqparams.txt",
                    f"--out={tmpbase}",
                    f"--fout={tmpbase}_field",
                    f"--iout={tmpbase}_b0u",
                    "--verbose"
        ]
        # prepare topup command with synb0 config options 
        if config is None:
            topup_cmd.extend(['--warpres=20,16,14,12,10,6,4',
                    '--subsamp=2,2,2,2,2,1,1',
                    '--fwhm=8,6,4,3,3,2,1',
                    '--miter=5,5,5,5,5,15,15',
                    '--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005',
                    '--ssqlambda=1',
                    '--regmod=bending_energy',
                    '--estmov=1,1,1,1,1,0,0',
                    '--minmet=0,0,0,0,0,1,1',
                    '--splineorder=3',
                    '--numprec=double',
                    '--interp=spline',
                    '--scale=1'
            ])
        else:
            tmp_config_file = os.path.join(tmpdir, os.path.basename(config))
            shutil.copyfile(config, tmp_config_file)
            topup_cmd.extend([f"--config={tmp_config_file}"])
        with open(output_base+'_command.txt', 'w') as fid:
            fid.write((' '.join(topup_cmd))+'\n')
        # run topup 
        logging.info("Run topup")
        ExecFSLCommand(topup_cmd).run()  
        # undo cropping/padding if necessary 
        shutil.copyfile(tmpbase+'_movpar.txt', output_base+'_movpar.txt')
        if adjusted1 or adjusted2:
            logging.debug('Adjusted')
            field_img = resample_image(read_nifti(tmpbase+'_field.nii.gz'), master=ref_img, interp='NearestNeighbor')
            field_img.to_filename(output_base+'_field.nii.gz')
            fieldcoef_img = resample_image(read_nifti(tmpbase+'_fieldcoef.nii.gz'), master=ref_img, interp='NearestNeighbor')
            fieldcoef_img.to_filename(output_base+'_fieldcoef.nii.gz')
            b0u_img = resample_image(read_nifti(tmpbase+'_b0u.nii.gz'), master=ref_img, interp='NearestNeighbor')
            b0u_img.to_filename(output_base+'_b0u.nii.gz')
            logging.debug(f'field_img.shape : {field_img.shape}')
            logging.debug(f'fieldcoef_img.shape : {fieldcoef_img.shape}')
            logging.debug(f'b0u_img.shape : {b0u_img.shape}')
        else:
            shutil.copyfile(tmpbase+'_field.nii.gz', output_base+'_field.nii.gz')
            shutil.copyfile(tmpbase+'_fieldcoef.nii.gz', output_base+'_fieldcoef.nii.gz')
            shutil.copyfile(tmpbase+'_b0u.nii.gz', output_base+'_b0u.nii.gz')
    return output_base

def apply_topup(images, topup_prefix, acqparamstxt, index=None):
    logging.debug('diciphr.diffusion.apply_topup')
    images = force_to_list(images)
    if len(images) < 1:
        raise DiciphrException('No images to apply topup to')
    if index is None:
        index = [1 for img in images]
    with TempDirManager(prefix='apply_topup') as manager:
        # Write images to file 
        tmpdir = manager.path()
        filenames = []
        outputf = os.path.join(tmpdir, 'output.nii.gz')
        for i, img in enumerate(images):
            outf = os.path.join(tmpdir, f"input{i}.nii.gz")
            write_nifti(outf, img)
            filenames.append(outf)
        cmd = ['applytopup', 
                '-i', ','.join(filenames), 
                '-a', acqparamstxt, 
                '-x', ','.join(map(str,index)), 
                '-t', topup_prefix, 
                '-o', outputf,
                '-m', 'jac']
        logging.info('Run applytopup')
        ExecFSLCommand(cmd).run()  
        result_img = read_nifti(outputf, lazy_load=False)
    return result_img 

def fsl_mcflirt(nifti_img):
    ''' 
    Run FSL mcflirt on a 4D image 
    ''' 
    with TempDirManager(prefix='fsl_eddy_correct') as manager:
        tmpdir = manager.path()
        input_filename = os.path.join(tmpdir, 'input.nii')
        out_filename = os.path.join(tmpdir, 'mcf.nii.gz')
        write_nifti(input_filename, nifti_img)
        cmd = ['mcflirt', '-in', input_filename, '-out', out_filename, '-refvol', '0']
        ExecFSLCommand(cmd).run()
        out_im = read_nifti(out_filename, lazy_load=False)
    return out_im
    
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
    with TempDirManager(prefix='fsl_eddy_correct') as manager:
        tmpdir = manager.path()
        dwi_tmp_filename=os.path.join(tmpdir,'dwi.nii')
        eddy_out_filebase=os.path.join(tmpdir,'output')
        eddy_out_filename=os.path.join(tmpdir,'output.nii.gz')
        dwi_im.to_filename(dwi_tmp_filename)
        bval_index = np.where(bvals == 0)[1][0] #first b0 image as reference. 
        cmd=['eddy_correct',dwi_tmp_filename,eddy_out_filebase,str(bval_index)]
        ExecFSLCommand(cmd).run()
        out_im = read_nifti(eddy_out_filename, lazy_load=False)
    return out_im
    
def fsl_eddy_rotate_bvecs(bvecs, eddy_parameters):
    # Code from scilpy by Maxime Descouteaux 
    logging.debug('diciphr.diffusion.fsl_eddy_rotate_bvecs')
    eddy_a = np.array(eddy_parameters)
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
    max_transl_error = np.max(norm_diff)
    max_rot_error = np.max(angle)*180/np.pi
    logging.info(f"{max_transl_error} mm is the maximum translation error")
    logging.info(f"{max_rot_error} degrees is the maximum rotation error")
    return bvecs_rotated.transpose()    
      
def fsl_eddy_post_topup(dwi_im, bvals, bvecs, topup_prefix, acqparamstxt, indextxt, mask_im, 
                        unwarped_b0_im=None, replace_outliers=True):
    eddy_exe = 'eddy'
    with TempDirManager(prefix='fsl_eddy_post_topup') as manager:
        tmpdir = manager.path()
        dwi_filename = os.path.join(tmpdir, 'dwi.nii.gz')
        bvals_filename = os.path.join(tmpdir, 'dwi.bval')
        bvecs_filename = os.path.join(tmpdir, 'dwi.bvec')
        mask_filename = os.path.join(tmpdir, 'mask.nii.gz')
        eddy_prefix = os.path.join(tmpdir, 'eddy_results')
        acqparams_tmp = os.path.join(tmpdir, 'acqparams.txt')
        index_tmp = os.path.join(tmpdir, 'index.txt')
        topup_tmp = os.path.join(tmpdir, 'topup')
        # ensure mask image grid and affine matches DWI image (Mask/DWI may have been padded to run topup)
        if not check_affines_and_shapes_match(mask_im, dwi_im):
            mask_im = resample_image(mask_im, interp='NearestNeighbor', master=dwi_im)
        if unwarped_b0_im is not None:
            # When running Synb0-Disco, data must be set to 0 where the undistorted image is 0, 
            # This occurs because of the registration back from MNI space. 
            # ensure image grid and affine matches DWI image (Image may have been padded to run topup)
            if not check_affines_and_shapes_match(unwarped_b0_im, dwi_im):
                unwarped_b0_im = resample_image(unwarped_b0_im, interp='NearestNeighbor', master=dwi_im)
            dwi_im = mask_dwi(dwi_im, threshold_image(unwarped_b0_im))
        
        # X-flip bvec, if data is LPS
#        orient_code = ''.join(nib.aff2axcodes(dwi_im.affine))
#        if orient_code == 'LPS':
#            bvecs = bvecs*np.array([-1,1,1])[...,None] 
        bvals = round_bvals(bvals)
        logging.debug('Write DWI and mask to tmpdir')
        write_dwi(dwi_filename, dwi_im, bvals, bvecs)
        write_nifti(mask_filename, mask_im)
        logging.debug('Copy acqparams and index to tmpdir')
        shutil.copyfile(acqparamstxt, acqparams_tmp)
        shutil.copyfile(indextxt, index_tmp)
        logging.debug('Copy topup files to tmpdir')
        shutil.copyfile(f"{topup_prefix}_fieldcoef.nii.gz",f"{topup_tmp}_fieldcoef.nii.gz")
        shutil.copyfile(f"{topup_prefix}_movpar.txt",f"{topup_tmp}_movpar.txt")
        numthreads = os.getenv('SLURM_CPUS_PER_TASK',os.getenv('OMP_NUM_THREADS','1'))
        eddy_cmd=[eddy_exe,
                    f"--imain={dwi_filename}",
                    f"--mask={mask_filename}",
                    f"--acqp={acqparams_tmp}",
                    f"--index={index_tmp}",
                    f"--bvecs={bvecs_filename}",
                    f"--bvals={bvals_filename}",
                    f"--out={eddy_prefix}",
                    f"--topup={topup_tmp}",
                    f"--nthr={numthreads}",
                    "--data_is_shelled",
                    "--very_verbose", 
                ]
        if replace_outliers: 
            eddy_cmd.append('--repol')
        
        ExecFSLCommand(eddy_cmd, environ={'OMP_NUM_THREADS':numthreads}).run()
        
        eddy_parameters = np.loadtxt('{}.eddy_parameters'.format(eddy_prefix))
        eddy_text_outputs = {
            'eddy_parameters': eddy_parameters,
            'eddy_movement_rms': np.loadtxt(f"{eddy_prefix}.eddy_movement_rms"),
            'eddy_restricted_movement_rms': np.loadtxt(f"{eddy_prefix}.eddy_restricted_movement_rms"),
            'eddy_outlier_map': np.loadtxt(f"{eddy_prefix}.eddy_outlier_map", skiprows=1),
            'eddy_outlier_n_stdev_map': np.loadtxt(f"{eddy_prefix}.eddy_outlier_n_stdev_map", skiprows=1),
            'eddy_outlier_n_sqr_stdev_map': np.loadtxt(f"{eddy_prefix}.eddy_outlier_n_sqr_stdev_map", skiprows=1)
        }
        dwi_eddy_im = read_nifti(f"{eddy_prefix}.nii.gz", lazy_load=False)
#        bvecs = fsl_eddy_rotate_bvecs(bvecs, eddy_parameters)
        bvecs = np.loadtxt(f"{eddy_prefix}.eddy_rotated_bvecs")
    # if a 3d volume bvecs will be shape (3,) and bvals will be shape (1,1)
    if len(bvecs.shape) == 1 and len(bvecs) == 3:
        bvecs = bvecs.reshape((3,1))      
#        if orient_code == 'LPS':
#            bvecs = bvecs*np.array([-1,1,1])[...,None] 
    return dwi_eddy_im, bvals, bvecs, eddy_text_outputs
    
def fsl_eddy(dwi_im, bvals, bvecs, mask_im, readout_time=0.062, replace_outliers=False):
    logging.debug('diciphr.diffusion.fsl_eddy')
    eddy_exe = 'eddy'
    with TempDirManager(prefix='fsl_eddy') as manager:
        tmpdir = manager.path()
        dwi_filename = os.path.join(tmpdir, 'dwi.nii.gz')
        bvals_filename = os.path.join(tmpdir, 'dwi.bval')
        bvecs_filename = os.path.join(tmpdir, 'dwi.bvec')
        mask_filename = os.path.join(tmpdir, 'mask.nii.gz')
        eddy_prefix = os.path.join(tmpdir, 'eddy_results')
        
        orient_code = ''.join(nib.aff2axcodes(dwi_im.affine))
#        if orient_code == 'LPS':
#            # flip X for LPS data to eddy 
#            bvecs = bvecs*np.array([-1,1,1])[...,None] 
        write_dwi(dwi_filename, dwi_im, bvals, bvecs)
        write_nifti(mask_filename, mask_im)
        indextxt = os.path.join(tmpdir, 'index.txt')
        acqparamstxt = os.path.join(tmpdir, 'acqparams.txt')
        nb_imgs = dwi_im.shape[-1]
        indx_list = [1 for i in range(nb_imgs)]
        with open(indextxt, 'w') as fid:
            fid.write(' '.join(map(str,indx_list)))
        with open(acqparamstxt, 'w') as fid:
            fid.write(' '.join(map(str,[0, 1, 0, readout_time])))
        numthreads = os.getenv('SLURM_CPUS_PER_TASK',os.getenv('OMP_NUM_THREADS','1'))
        cmd=[eddy_exe,
                f'--imain={dwi_filename}',
                f'--mask={mask_filename}',
                f'--acqp={acqparamstxt}',
                f'--index={indextxt}',
                f'--bvecs={bvecs_filename}',
                f'--bvals={bvals_filename}',
                f'--out={eddy_prefix}',
                f'--nthr={numthreads}',
                '--very_verbose',
                '--data_is_shelled',
            ]
        if replace_outliers:
            cmd.extend(['--repol'])
        ExecFSLCommand(cmd, environ={'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS',1)}).run()
        eddy_parameters = np.loadtxt('{}.eddy_parameters'.format(eddy_prefix))
        eddy_text_outputs = {
            'eddy_parameters': eddy_parameters,
            'eddy_movement_rms': np.loadtxt(f'{eddy_prefix}.eddy_movement_rms'),
            'eddy_restricted_movement_rms': np.loadtxt(f'{eddy_prefix}.eddy_restricted_movement_rms'),
            'eddy_outlier_map': np.loadtxt(f'{eddy_prefix}.eddy_outlier_map', skiprows=1),
            'eddy_outlier_n_stdev_map': np.loadtxt(f'{eddy_prefix}.eddy_outlier_n_stdev_map', skiprows=1),
            'eddy_outlier_n_sqr_stdev_map': np.loadtxt(f'{eddy_prefix}.eddy_outlier_n_sqr_stdev_map', skiprows=1)
        }
        dwi_eddy_im = read_nifti('{}.nii.gz'.format(eddy_prefix), lazy_load=False)
        # bvecs = fsl_eddy_rotate_bvecs(bvecs, eddy_parameters)
        bvecs = np.loadtxt(f"{eddy_prefix}.eddy_rotated_bvecs")
        if orient_code == 'LPS':
            bvecs = bvecs*np.array([-1,1,1])[...,None] 
    return dwi_eddy_im, bvals, bvecs, eddy_text_outputs

def save_eddy_text(prefix, eddy_text_outputs):
    np.savetxt(prefix+'_eddy_parameters.txt', eddy_text_outputs['eddy_parameters'], delimiter=' ', fmt='%0.12f')
    np.savetxt(prefix+'_eddy_movement_rms.txt', eddy_text_outputs['eddy_movement_rms'], delimiter=' ', fmt='%0.12f')
    np.savetxt(prefix+'_eddy_restricted_movement_rms.txt', eddy_text_outputs['eddy_restricted_movement_rms'], delimiter=' ', fmt='%0.12f')
    np.savetxt(prefix+'_eddy_outlier_map.txt', eddy_text_outputs['eddy_outlier_map'], delimiter=' ', fmt='%d')
    np.savetxt(prefix+'_eddy_outlier_n_stdev_map.txt', eddy_text_outputs['eddy_outlier_n_stdev_map'], delimiter=' ', fmt='%0.6f')
    np.savetxt(prefix+'_eddy_outlier_n_sqr_stdev_map.txt', eddy_text_outputs['eddy_outlier_n_sqr_stdev_map'], delimiter=' ', fmt='%0.6f')      

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
        brain_mask_im = nifti_image((fa_im.get_fdata() > 0).astype(np.int16), fa_im.affine)
    if erode_iterations > 0:
        brain_mask_ero_im = erode_image(brain_mask_im, iterations=erode_iterations)
    fa_data = fa_im.get_fdata()
    brain_mask_ero_data = brain_mask_ero_im.get_fdata()
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
def n4_bias_correct_dwi(dwi_img, bvals, bvecs, field=False, mask_img=None, weight_img=None,
                convergence=None, bspline=None):
    ''' 
    Run ANTs N4BiasFieldCorrection on a DWI image.
    '''
    if convergence is None:
        convergence = '[50x50x50x50,1e-3]'
    if bspline is None:
        bspline = '[150,3]'
    with TempDirManager(prefix='n4biasdwi') as manager:
        tmpdir = manager.path()
        b0_filename = os.path.join(tmpdir, 'b0.nii')
        outimg_fn = os.path.join(tmpdir, 'output.nii')
        outfield_fn = os.path.join(tmpdir, 'field.nii')
        # extract b0 
        b0_im = extract_b0(dwi_img, bvals)
        b0_data = b0_im.get_fdata()
        percLow = np.percentile(b0_data[b0_data>0],0.5)
        percHigh = np.percentile(b0_data[b0_data>0],99.5)
        b0_winsor = b0_data.copy()
        b0_winsor[b0_winsor < percLow] = percLow
        b0_winsor[b0_winsor < percHigh] = percHigh
        write_nifti(b0_filename, nifti_image(b0_winsor, b0_im.affine))
        cmd = [ 'N4BiasFieldCorrection', '-d', '3', '-i', b0_filename,
                '-r', '-b', str(bspline),'-c', str(convergence), '-v', '1',
                '-o', f'[{outimg_fn},{outfield_fn}]'
        ]
        if mask_img is not None:
            mask_filename = os.path.join(tmpdir, 'mask.nii')
            write_nifti(mask_filename, mask_img)
            cmd += ['-x', mask_filename]
        if weight_img is not None:
            weight_filename = os.path.join(tmpdir, 'weight.nii')
            write_nifti(weight_filename, weight_img)
            cmd += ['-w',weight_filename]
        ExecCommand(cmd).run()
        field_im = read_nifti(outfield_fn, lazy_load=False)
        # Correct each volume of the dwi 
        dwi_data = dwi_img.get_fdata()
        field_data = field_im.get_fdata()
        logging.info('Divide DWI image every volume by bias field map')
        dwi_corr = dwi_data / field_data[...,None]
        dwi_corr_im = nifti_image(dwi_corr, dwi_img.affine)
    if field:
        return (dwi_corr_im, bvals, bvecs), field_im 
    else:
        return dwi_corr_im, bvals, bvecs

def dwi_multi_b0_temporal_snr(dwi_im, bvals, bvecs, mask_im):
    tol=20
    b0_indices = np.where(bvals.flatten() < tol)
    if len(b0_indices[0]) < 3:
        raise DiciphrException('Cannot calculate temporal SNR with too few b-zero images.')
    data=dwi_im.get_fdata()
    affine=dwi_im.affine
    mask_data=mask_im.get_fdata()
    b0s=data[:,:,:,b0_indices]
    std=np.std(b0s, axis=-1)
    mn=np.mean(b0s, axis=-1)
    tsnr=mn/std
    tsnr[np.logical_not(mask_data)] = 0
    tsnr_img = nifti_image(tsnr, affine)
    return tsnr_img
    
def dti_roi_stats(tensor_im, atlas_im, labels=None, scalars=['FA','TR','AX','RAD'], measures=['mean','median','std'], nonzero=True, min_roi_size=20, mask_im=None):
    logging.debug('diciphr.diffusion.dti_roi_stats')
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
    atlas_data = atlas_im.get_fdata()
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
    scalar_data = scalar_im.get_fdata()
    atlas_data = atlas_im.get_fdata().astype(int)
    if mask_im is not None:
        mask = mask_im.get_fdata() > 0 
    else:
        mask = atlas_data > 0
    if nonzero:
        mask = np.logical_and(mask, scalar_data != 0)
    if labels is None:
        labels = range(1,atlas_data.max()+1)
    num_labels = len(labels)
    atlas_volumes=[]
    scalar_data = scalar_im.get_fdata().astype(np.float32)[mask]
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
    motion_table = np.loadtxt(eddy_params_filename)[:,:6]
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
