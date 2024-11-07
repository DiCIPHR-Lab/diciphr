# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, sys, shutil, logging, traceback
import nibabel as nib
import numpy as np
import scipy.ndimage
from collections import OrderedDict
from .utils import ( make_dir, make_temp_dir, which, 
                ExecCommand, DiciphrException, logical_and, force_to_list )

##############################################
############   NIFTI FUNCTIONS  ##############
##############################################
def is_nifti_file(filename):
    """Returns True if filename is a nifti file. """
    try:
        im = nib.Nifti1Image.from_filename(filename)
    except:
        return False
    return True

def strip_ext(filename):
    '''Strips a generic filename off a path. If extension is gz or bz2, will try a second round.'''
    ext1 = filename[::-1].split('.')[0][::-1]
    if ext1 in ['gz','bz2']:
        ext2 = filename[::-1].split('.')[1][::-1]
        ext1 = ".".join((ext2,ext1))
    return filename[:-(len(ext1)+1)]
    
def strip_nifti_ext(filename):
    '''Strips the extension off a nifti path and returns the rest.'''
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    elif filename[-4:] in ['.nii','.hdr','.img']:
        return filename[:-4]
    else:
        raise DiciphrException('Filename {} is not a Nifti path!'.format(filename))

def has_nifti_ext(filename):
    '''Returns true if the file ends with .hdr, .img, .nii, or .nii.gz .'''
    try:
        strip_nifti_ext(filename)
    except DiciphrException:
        return False
    return True
    
def get_nifti_prefix(filename):
    '''Strip the nifti extension off a filename and return the basename.'''
    return strip_nifti_ext(os.path.basename(filename))
    
def get_nifti_ext(filename):
    '''Strip the nifti extension off a filename and return it.'''
    if filename.endswith('.nii.gz'):
        return 'nii.gz'
    elif filename [-4:] in ('.hdr','.img','.nii'):
        return filename[-3:]
    else:
        raise DiciphrException('Filename {} is not a Nifti path!'.format(filename))
        
def find_nifti_from_basename(prefix):
    '''Find a nifti file by adding a nifti extension to a prefix. If cannot be found, raise a DiciphrException.'''
    found = []
    for ext in ['hdr','nii','nii.gz']:
        filename='.'.join((prefix,ext))
        try:
            os.stat(filename)
            if not is_nifti_file(filename):
                raise DiciphrException('Encountered invalid nifti-like file {}'.format(prefix))
            else:
                found.append(filename)
        except: 
            pass
    if len(found) == 0:
        raise DiciphrException('Cannot find nifti file from prefix {}'.format(prefix))
    if len(found) > 1: 
        raise DiciphrException('Found multiple nifti files from prefix {}'.format(prefix))
    return found[0]
    
def nifti_image(data, affine, header=None, dtype=None, intent=0, **kwargs):
    '''Create a Nifti1Image instance. Casts data to a dtype if given.
    
    Parameters
    ----------
    data : numpy.ndarray
        The image data 
    affine : numpy.ndarray
        The affine transformation from voxel indices to real world coordinates.
    header : Optional[nibabel.Nifti1Header]
        The nifti header dictionary
    dtype : Optional[type]
        If provided, casts the data to this datatype and updates the header.
        
    Returns 
    -------
    nibabel.Nifti1Image
        The nifti image instance
    '''
    if dtype is not None:
        data=data.astype(dtype)
    if data.dtype == np.dtype('bool'):
        data = data.astype(np.int16)
    im=nib.Nifti1Image(data, affine, header=header)
    im.header.set_sform(affine)
    im.header.set_qform(affine)
    im.header.set_data_dtype(data.dtype)
    im.header.set_intent(intent)
    for key in kwargs:
        im.header[key] = kwargs[key]
    im.update_header()
    return im
    
def read_nifti(filename, dtype=None):
    '''Wrapper function for nibabel.load'''
    logging.debug('diciphr.utils.read_nifti')
    im=nib.load(filename)
    if dtype:
        data = im.get_data().astype(dtype)
        im = nib.Nifti1Image(data, im.affine, im.header)
        im.header.set_data_dtype(dtype)
    return im
    
def write_nifti(filename,nifti_im=None,data=None,affine=None):
    '''Wrapper function for nibabel.save
    
    Parameters 
    ----------
    filename : str
        Output filename
    nifti_im : Optional[nibabel.Nifti1Image]
        A nibabel Nifti1Image object.
    data : Optional[numpy.ndarray]
        A numpy array of image data. 
        To use, nifti_im should be None, and affine is required. 
    affine : Optional[numpy.ndarray]
        An affine transformation for nifti header. 
        
    Returns
    -------
    str
        The output filename
    '''
    logging.debug('diciphr.utils.write_nifti')
    if not has_nifti_ext(filename):
        raise DiciphrException('Filename {} is not a valid Nifti path!'.format(filename))
    if nifti_im is None:
        nifti_im = nifti_image(data, affine)
    nifti_im.to_filename(filename)
    logging.info("Writing NiFTI to file {}".format(filename))
    return filename
        
def read_gradients(bval_file, bvec_file):
    '''Read bval and bvec files.
    
    Parameters 
    ----------
    bval_file : str
        Path to bval text file. If not provided, path will be inferred from the bvec filename.
    bvec_file : str
        Path to bval text file. If not provided, path will be inferred from the bval filename.
        
    Returns
    -------
    tuple
        A tuple of bvals, bvecs (numpy.ndarray)
    '''
    bvals = np.loadtxt(bval_file)
    bvals = bvals.reshape((1,np.size(bvals)))                       
    bvecs = np.loadtxt(bvec_file)
    return bvals, bvecs

def is_valid_dwi(dwi_im, bvals, bvecs, raise_if_invalid=False):
    '''Check that a DWI image and associated bvals and bvecs are valid.
    
    DWI time dimension (N) must match bvals and bvecs 2nd dimensions. 
    bvals must be of shape (1,N)
    bvecs must be of shape (3,N)
    bvals should contain zeros, and where bvals is equal to zero, bvecs must also be equal to zero. 
    If bvals does not contain zeros, gives a warning.

    Parameters 
    ----------
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
    raise_if_invalid : Optional[bool]
        raise a DiciphrException if dwi is not valid.
        
    Returns
    -------
    bool
    '''
    bvecs_n=bvecs.shape[1]
    bvecs_dim=bvecs.shape[0]
    bvals_n=bvals.shape[1]
    bvals_dim=bvals.shape[0]
    dwi_n=dwi_im.shape[-1]
    dimensions_valid = (dwi_n == bvals_n) and (dwi_n == bvecs_n) and (bvecs_dim == 3) and (bvals_dim == 1)
    ret = dimensions_valid
    if raise_if_invalid and not ret:
        raise DiciphrException('Not a valid diffusion nifti and bval/bvec')
    return ret
    
def read_dwi(filename, bval_file=None, bvec_file=None):
    '''Read a nifti file and associated bval and bvec files.
    
    Parameters 
    ----------
    filename : str
        Path to diffusion weighted image NiFTI
    bval_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
    bvec_file : Optional[str]
        Path to bval text file. If not provided, path will be inferred from the dwi filename.
        
    Returns
    -------
    tuple
        A tuple of dwi_im (nibabel.Nifti1Image), bvals, bvecs (numpy.ndarray)
    '''
    logging.debug('diciphr.diffusion.read_dwi')
    dwi_im = nib.load(filename)
    if bval_file is None:
        bval_file = strip_nifti_ext(filename)+'.bval'
    if bvec_file is None:
        bvec_file = strip_nifti_ext(filename)+'.bvec'
    if not (os.path.exists(bval_file) and os.path.exists(bvec_file)):
        raise DiciphrException('Cannot find bval/bvec files for DWI image')
    else:
        bvals, bvecs = read_gradients(bval_file, bvec_file) 
    is_valid_dwi(dwi_im, bvals, bvecs, True)
    return dwi_im, bvals, bvecs
    
def write_dwi(filename,dwi_im,bvals,bvecs):
    '''Write a DWI NiFTI image and associated bvals and bvecs to file.
    
    Parameters 
    ----------
    filename : str
        The output nifti path with file extension.
    dwi_im : nibabel.Nifti1Image
        The DWI NiFTI image
    bvals : numpy.ndarray
        The bvals array
    bvecs : numpy.ndarray
        The bvecs array
        
    Returns
    -------
    None
    '''
    logging.debug('diciphr.diffusion.write_dwi')
    is_valid_dwi(dwi_im, bvals, bvecs, True)
    write_nifti(filename, dwi_im)
    bvalfile=strip_nifti_ext(filename)+'.bval'
    bvecfile=strip_nifti_ext(filename)+'.bvec'
    np.savetxt(bvalfile,bvals,fmt="%0.1f")
    np.savetxt(bvecfile,bvecs,fmt="%0.8f")
    
def intersection_mask(nifti_files):
    return nifti_image(logical_and(*[nib.load(f).get_data() > 0])*1, nib.load(nifti_files)[0].affine)
    
###############################################
############  NIFTI Orientation  ##############
###############################################
def _get_transformation_ornt(orig_orient_code, new_orient_code):
    new_orient_ornt = nib.orientations.axcodes2ornt(new_orient_code)
    orig_orient_ornt = nib.orientations.axcodes2ornt(orig_orient_code)
    trans_ornt = nib.orientations.ornt_transform(orig_orient_ornt,
                                                  new_orient_ornt)
    return trans_ornt
    
def reorient_nifti(nifti_im, orientation = 'LPS'):
    
    ''' Reorient the image from original orientation in nii to a given orientation. 
    
        :param nii: A nifti image to be reoriented.
        :type nii: nibabel.nifti1.Nifti1Image
        :param orientation: A nifti image orientation in a string using CBICA "to" convention.
        :type orientation: str
        :param bvec: A 3 x N array of gradient vectors.
        :type bvec: numpy.array
        :returns: nifti_im_reoriented
    '''
    if not _is_valid_orientation(orientation):
        raise DiciphrException('Not a valid orientation: {}'.format(orientation))
    
    nii_affine = nifti_im.affine
    nii_data = nifti_im.get_data()
    new_orient_code = tuple(s.upper() for s in orientation)
    orig_orient_code = nib.aff2axcodes(nii_affine)
    trans_ornt = _get_transformation_ornt(orig_orient_code, new_orient_code)
    trans_affine = nib.orientations.inv_ornt_aff(trans_ornt, nifti_im.shape)
    nii_data_reoriented = nib.apply_orientation(nii_data, trans_ornt)
    nii_affine_reoriented = np.dot(nii_affine, trans_affine)
    nifti_im_reoriented = nib.Nifti1Image(nii_data_reoriented,
                                      nii_affine_reoriented,
                                      header = nifti_im.header)
    nifti_im_reoriented.set_qform(nifti_im_reoriented.get_sform())
    return nifti_im_reoriented
    
def reorient_bvec(bvec, old_orientation, new_orientation): 
    '''Reorient the bvec from original orientation to a given orientation. 
    
    Parameters
    ----------
    bvec : numpy.ndarray
        A bvec numpy array 
    old_orientation : str
        An orientation string or tuple of 3 characters
    new_orientation : str
        An orientation string or tuple of 3 characters
        
    Returns
    -------
    numpy.ndarray
        The reoriented bvec array
    '''
    if not _is_valid_orientation(old_orientation):
        raise DiciphrException('Not a valid orientation: {}'.format(old_orientation))
    if not _is_valid_orientation(new_orientation):
        raise DiciphrException('Not a valid orientation: {}'.format(new_orientation))
    orig_orient_code = tuple(s.upper() for s in old_orientation)
    new_orient_code = tuple(s.upper() for s in new_orientation)
    orig_orient_ornt = nib.orientations.axcodes2ornt(orig_orient_code)
    new_orient_ornt = nib.orientations.axcodes2ornt(new_orient_code)
    trans_ornt = nib.orientations.ornt_transform(orig_orient_ornt,
                                                  new_orient_ornt)
    trans_affine = nib.orientations.inv_ornt_aff(trans_ornt, (128,128,80))  #dummy nifti shape
    bvec = np.array(bvec).T
    bvec_out = np.dot(bvec, trans_affine[0:3,0:3])
    bvec_out = bvec_out.T
    return bvec_out
    
def reorient_dwi(dwi_im, bvals, bvecs, orientation = 'LPS'):
    ''' Reorient the DWI image from original orientation in nii to a given orientation. 
    
        :param nii: A nifti dwi image to be reoriented.
        :type nii: nibabel.nifti1.Nifti1Image
        :param orientation: A nifti image orientation in a string using CBICA "to" convention.
        :type orientation: str
        :param bvec: A 3 x N array of gradient vectors.
        :type bvec: numpy.array
        :returns: (dwi_im_reoriented, bvals, bvecs_reoriented)
    '''
    if not _is_valid_orientation(orientation):
        raise DiciphrException('Not a valid orientation: {}'.format(orientation))
    nii_affine = dwi_im.affine
    new_orient_code = tuple(s.upper() for s in orientation)
    orig_orient_code = nib.aff2axcodes(nii_affine)
    dwi_im_reoriented = reorient_nifti(dwi_im, orientation)
    bvecs_reoriented = reorient_bvec(bvecs, orig_orient_code, new_orient_code)
    return (dwi_im_reoriented, bvals, bvecs_reoriented)       
    
def get_nifti_orientation(nifti_im):
    ''' Return the 3-letter code for orientation of a nifti image.'''
    return ''.join(nib.aff2axcodes(nifti_im.affine)).upper()

def _is_valid_orientation(orientation):
    ''' Check if a nifti orientation is valid. The orientation has to have three unique letters
        for three dimensional data, one for each dimension: left-right, anterior-posterior and
        superior-inferior.
        
        :param orientation: A nifti image orientation using CBICA "to" convention.
        :type orientation: str
        :returns: True if input string is a valid orientation.
    '''
    orientation = tuple(s.upper() for s in orientation)
    is_valid = True
    # if the string is shorter than three, fail it.
    if len(''.join(orientation)) != 3:
        is_valid = False
    
    LR = False
    AP = False
    SI = False
    LRnotTwice = True
    APnotTwice = True
    SInotTwice = True
    
    for o in orientation:
        if o in ('R','L'):
            if LR:
                LRnotTwice = False
            LR = True
        elif o in ('A','P'):
            if AP:
                APnotTwice = False
            AP = True
        elif o in ('S','I'):
            if SI:
                SInotTwice = False
            SI = True
    is_valid = (LR and AP and SI and LRnotTwice and APnotTwice and SInotTwice and is_valid)
    return is_valid
        
#############################################
#######  Resampling and Morphology  #########
#############################################   
def resample_image(nifti_im, voxelsizes=None, interp='Linear', master=None, workdir=None):
    '''Resample a nifti image with ANTs.
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be resampled
    voxelsizes : Optional[tuple]
        The target voxel dimensions, a tuple of length 3.
    interp: Optional[str]
        Interpolation mode, one of 'Linear','NearestNeighbor','MultiLabel','Gaussian','BSpline','GenericLabel',
                            'CosineWindowedSinc','WelchWindowedSinc','HammingWindowedSinc','LanczosWindowedSinc'
    master: Optional[nibabel.Nifti1Image]
        A nifti image to which align input dataset grid 
    Returns
    -------
    nibabel.Nifti1Image
        The resampled Nifti image
    '''
    logging.debug('diciphr.nifti_utils.resample_image')
    resample_exe = which(os.path.join(os.environ.get('ANTSPATH'), 'ResampleImage'))
    applyxfm_exe = which(os.path.join(os.environ.get('ANTSPATH'), 'antsApplyTransforms'))
    allowed_interps = ['Linear','NearestNeighbor','MultiLabel','Gaussian','BSpline','GenericLabel',
        'CosineWindowedSinc','WelchWindowedSinc','HammingWindowedSinc','LanczosWindowedSinc']
    if interp not in allowed_interps:   
        raise DiciphrException('interp must be one of {}.'.format(' '.join(allowed_interps)))
    if voxelsizes is None and master is None:
        raise DiciphrException('One of voxelsizes or master must be provided')
    # tensor input - need to implement log-Euclidean interpolation. 
    if nifti_im.header['intent_code'] == 1005:
        raise DiciphrException('Tensor input not enabled')
    # 4d input - split into 3d and resample each 
    if len(nifti_im.header.get_zooms()) == 4:
        split_images = split_image(nifti_im, dimension='t')
        resampled_images = []
        for im in split_images:
            res = resample_image(im, voxelsizes=voxelsizes, interp=interp, master=master)
            resampled_images.append(res)
        result_im = concatenate_niftis(*resampled_images)
    elif len(nifti_im.header.get_zooms()) == 3:
        if workdir is None:
            tmpdir = make_temp_dir(prefix='resample')
        else:
            tmpdir = workdir 
            os.makedirs(tmpdir, exist_ok=True)
        nifti_im.to_filename(os.path.join(tmpdir, 'input.nii.gz'))
        try:
            if master is None:
                # 3d input - without master_im 
                # create template image 
                resample_cmd = [resample_exe, '3', os.path.join(tmpdir, 'input.nii.gz'), 
                    os.path.join(tmpdir, 'regrid.nii.gz'), str(voxelsizes[0])+'x'+str(voxelsizes[1])+'x'+str(voxelsizes[2])]
                ExecCommand(resample_cmd).run()
                regrid_im = nib.load(os.path.join(tmpdir, 'regrid.nii.gz'))
                regrid_im = nifti_image(regrid_im.get_fdata(), regrid_im.affine, regrid_im.header)
                result_im = resample_image(nifti_im, interp=interp, master=regrid_im, workdir=tmpdir)
            else:
                # 3d input - with master 
                master.to_filename(os.path.join(tmpdir, 'ref.nii.gz'))
                resample_cmd = [applyxfm_exe, '-d', '3', '-i', os.path.join(tmpdir, 'input.nii.gz'),
                        '-o', os.path.join(tmpdir, 'output.nii.gz'), '-r', os.path.join(tmpdir, 'ref.nii.gz'),
                        '-n', interp, '--float','-v','1']
                ExecCommand(resample_cmd).run()
                result_im = nib.load(os.path.join(tmpdir, 'output.nii.gz'))
                result_im = nifti_image(result_im.get_fdata(), result_im.affine, result_im.header)
        finally:
            if workdir is None:
                shutil.rmtree(tmpdir)
    return result_im 
    
def erode_image(nifti_im, kernel='NN1', kernel_size=1, iterations=1, binarize=True, erode_away_from_border=True):
    '''Erode a nifti image with scipy or with fsl.
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be eroded
    kernel : Optional[str] 
        The type of kernel, choose from ['NN1','NN2','NN3','sphere','gauss'
    kernel_size: Optional[int]
        The size of the kernel. 
        Defaults to 1 for the box kernels (NN1, NN2, NN3) ;
        1.5 mm for the "gauss" and "sphere" kernels ;
        Must be a positive odd integer for the box kernels.
    iterations: Optional[int]
        Number of times to iterate the erosion.
    binarize: Optional[bool]
        If True (default), returns a mask where foreground is 1 and background is 0. 
    erode_away_from_border: Optional[bool]
        If True (default), voxels touching border will be eroded away. 
    Returns
    -------
    nibabel.Nifti1Image
        The eroded mask image
    '''
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
    def _is_odd_pos(integer):
        _is_num = (float(integer) == integer)
        _is_int = (int(integer) == integer)
        _is_pos = (integer > 0) 
        _is_odd = (integer % 2 == 1) 
        return _is_num and _is_int and _is_pos and _is_odd
        
    if not kernel in ['NN1','NN2','NN3','sphere','gauss','box']:
        raise DiciphrException('Kernel {} not supported'.format(kernel))
    
    #pad by 1 voxel so that voxels touching image boundary are eroded. 
    if erode_away_from_border: 
        nifti_im = crop_pad_image(nifti_im, x_adjust=[1,1], y_adjust=[1,1], z_adjust=[1,1])
    data = nifti_im.get_data()
    affine = nifti_im.affine
    binarized_data = np.zeros(data.shape, dtype='i4')
    binarized_data[data > 0] = 1 
    if kernel in ['NN1','NN2','NN3']:
        if not _is_odd_pos(kernel_size):
            raise DiciphrException('Kernel size needs to be a positive odd integer')
        nn_kernel = generate_binary_structure(3,int(kernel[-1]))
        eroded_data = binary_erosion(binarized_data, nn_kernel, iterations=iterations).astype(data.dtype)
    elif kernel in ['sphere','gauss','box']:
        if kernel == 'box' and not _is_odd_pos(kernel_size):
            raise DiciphrException('Kernel size needs to be a positive odd integer')
        tmpdir = make_temp_dir(prefix='fsl_erosion')
        tmp_nii = os.path.join(tmpdir,'in.nii')
        tmp_ero_out = os.path.join(tmpdir,'out.nii')
        input_im = nib.Nifti1Image(binarized_data, affine)
        input_im.to_filename(tmp_nii)
        cmd = ['fslmaths',tmp_nii,'-kernel',kernel,str(kernel_size),'-ero',tmp_ero_out]
        ExecCommand(cmd,environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        tmp_im = nib.load(tmp_ero_out)
        eroded_data = tmp_im.get_data()
    if not binarize:
        eroded_data = eroded_data * data
    else:
        eroded_data = eroded_data.astype('i4')
    out_im = nib.Nifti1Image(eroded_data, affine)
    out_im.header.set_sform(affine)
    out_im.header.set_qform(affine)
    out_im.update_header()
    if erode_away_from_border:
        out_im = crop_pad_image(out_im, x_adjust=[-1,-1], y_adjust=[-1,-1], z_adjust=[-1,-1])
    return out_im
    
def dilate_image(nifti_im, kernel='NN1', kernel_size=1, iterations=1, binarize=True, mode_dilation=True):
    '''Erode a nifti image with scipy or with fsl.
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be eroded
    kernel : Optional[str] 
        The type of kernel, choose from ['NN1','NN2','NN3','sphere','gauss'
    kernel_size: Optional[int]
        The size of the kernel. 
        Defaults to 1 for the box kernels (NN1, NN2, NN3) and the "gauss" and "sphere" kernels ;
        Must be a positive odd integer for the box kernels.
    iterations: Optional[int]
        Number of times to iterate the erosion.
    binarize: Optional[bool]
        If true, returns a mask where foreground is 1 and background is 0. 
    mode_dilation: Optional[bool]
        If true, and binarize is false, will use modal dilation ("dilD") else use mean dilation ("dilM")
        
    Returns
    -------
    nibabel.Nifti1Image
        The eroded mask image
    '''
    from scipy.ndimage.morphology import generate_binary_structure, binary_dilation
    def _is_odd_pos(integer):
        _is_num = (float(integer) == integer)
        _is_int = (int(integer) == integer)
        _is_pos = (integer > 0) 
        _is_odd = (integer % 2 == 1) 
        return _is_num and _is_int and _is_pos and _is_odd
        
    if not kernel in ['NN1','NN2','NN3','sphere','gauss','box']:
        raise DiciphrException('Kernel {} not supported'.format(kernel))
    if kernel in ['NN1','NN2','NN3']:
        if not _is_odd_pos(kernel_size):
            raise DiciphrException('Kernel size needs to be a positive odd integer')
        nn_kernel = generate_binary_structure(3,int(kernel[-1]))
        data = nifti_im.get_data()
        binarized_data = np.zeros(data.shape, dtype=np.int8)
        binarized_data[data > 0] = 1 
        _dtype = nifti_im.header.get_data_dtype()
        dilated_data = binary_dilation(binarized_data, nn_kernel, iterations=iterations).astype(_dtype)
        out_im = nib.Nifti1Image(dilated_data, nifti_im.affine)
    elif kernel in ['sphere','gauss','box']:
        dil_mode='-dilM'
        if mode_dilation and not binarize:
            dil_mode = '-dilD'
        tmpdir = make_temp_dir(prefix='fsl_dilation')
        tmp_nii = os.path.join(tmpdir,'in.nii')
        tmp_dil_out = os.path.join(tmpdir,'out.nii')
        nifti_im.to_filename(tmp_nii)
        cmd = ['fslmaths',tmp_nii,'-kernel',kernel,str(kernel_size),dil_mode,tmp_dil_out]
        ExecCommand(cmd,environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
        tmp_im = nib.load(tmp_dil_out)
        out_im = nib.Nifti1Image(tmp_im.get_data(), tmp_im.affine, tmp_im.header)
    return out_im
  
def cut_region(roi_img, k):
    '''Cuts a region into k sub-regions.
    
    Parameters
    ----------
    roi_im : nibabel.Nifti1Image
        Input region to cluster
    k : int 
        The number of expected sub-regions
    Returns
    -------
    list
        A list of nibabel.Nifti1Image objects 
    '''
    from scipy.cluster.vq import kmeans2
    affine = roi_img.get_affine()
    roi = roi_img.get_data()
    logging.info("Perform clustering...")
    indices = np.array(np.nonzero(roi), dtype=np.float64).T
    codebook, label = kmeans2(indices, k)
    hdr = nib.Nifti1Header()
    hdr.set_qform(affine)
    hdr.set_sform(affine)
    mask_imgs = []
    for i in range(k):
        mask = np.zeros(roi.shape, dtype=np.uint8)
        indices_subregion = np.array(indices[label == i], dtype=int)
        for j in range(indices_subregion.shape[0]):
            mask[indices_subregion[j, 0],
                 indices_subregion[j, 1],
                 indices_subregion[j, 2]] = 1
        mask_imgs.append(nifti_image(mask, affine))
    return mask_imgs
    
############################################
############  Other Utilities  #############
############################################    
def threshold_image(nifti_im, threshold_value):
    logging.info('Calculating mask with threshold at {}.'.format(threshold_value))
    thresh_data = (nifti_im.get_data() > float(threshold_value)).astype(np.int16)
    mask_im = nib.Nifti1Image(thresh_data, nifti_im.affine)
    return mask_im 
    
def flirt_register(input_im, ref_im, 
                    cost='normmi', searchcost='normmi', dof=12, coarsesearch=60, finesearch=18, 
                    search_range=[], searchrx=[-90,90], searchry=[-90,90], searchrz=[-90,90], 
                    interp='trilinear', return_mat=True, return_im=True):
    ''' Runs flirt registration from an input image to a reference image.
    
    Parameters
    ----------
        input_im : nibabel.Nifti1Image
            The image to be registered
        ref_im : nibabel.Nifti1Image
            The reference image
        cost : str
        searchcost : str
        dof : int
        coarsesearch : int
        finesearch : int
        search_range : list
            If provided, sets "searchrx", "searchry", "searchrz" options to the same values.
        searchrx : list
        searchrx : list
        searchrx : list
        interp : str
        return_mat : bool
            If True, returns the resulting affine matrix
        return_im : bool
            If True, returns the resulting nifti image
    '''
    
    if not cost in ('mutualinfo','corratio','normcorr','normmi','leastsq','labeldiff'):
        raise DiciphrException('cost argument must be one of mutualinfo, corratio, normcorr, normmi, leastsq, labeldiff')
    if not searchcost in ('mutualinfo','corratio','normcorr','normmi','leastsq','labeldiff'):
        raise DiciphrException('searchcost argument must be one of mutualinfo, corratio, normcorr, normmi, leastsq, labeldiff')
    if not interp in ('trilinear','nearestneighbor','nearestneighbour','sinc'):
        raise DiciphrException('interp argument must be one of trilinear, nearestneighbour, sinc')
    if interp == 'nearestneighbor':
        interp = 'nearestneighbour'
    if not dof in (3, 6, 7, 12):
        raise DiciphrException('dof argument must be one of 3, 6, 7, 12')
    try:
        coarsesearch == float(coarsesearch)
        if coarsesearch <= 0: 
            raise ValueError('non-positive') 
    except:
        raise DiciphrException('coarsesearch argument must be a positive number!')
    try:
        finesearch == float(finesearch)
        if finesearch <= 0: 
            raise ValueError('non-positive') 
    except:
        raise DiciphrException('finesearch argument must be a positive number!')
    if search_range:
        searchrx = search_range
        searchry = search_range
        searchrz = search_range
    search_ranges= [searchrx, searchry, searchrz]
    for idx in range(3):
        try:
            _s1, _s2 = search_ranges[idx]
            _s1 == float(_s1)
            _s2 == float(_s2)
            search_ranges[idx] = (_s1, _s2)
        except:
            raise DiciphrException('search range must be a list of two numbers') 
    #begin
    tmpdir = make_temp_dir(prefix='fsl_flirt')
    input_filename = os.path.join(tmpdir, 'input.nii')
    ref_filename = os.path.join(tmpdir, 'ref.nii')
    outmat_filename = os.path.join(tmpdir, 'output.mat')
    output_filename = os.path.join(tmpdir, 'output.nii')
    logging.debug('Writing inputs to tmpdir {}'.format(tmpdir))
    input_im.to_filename(input_filename)
    ref_im.to_filename(ref_filename)
    cmd = ['flirt', '-in', input_filename, '-ref', ref_filename, 
           '-omat', outmat_filename, '-out', output_filename, 
           '-cost', cost, '-searchcost', searchcost, '-dof', dof, '-interp', interp, 
           '-coarsesearch', coarsesearch, '-finesearch', finesearch, 
           '-searchrx', search_ranges[0][0], search_ranges[0][1],
           '-searchrx', search_ranges[1][0], search_ranges[1][1], 
           '-searchrx', search_ranges[2][0], search_ranges[2][1]]
    cmd = list(map(str,cmd)) 
    logging.debug(' '.join(cmd))
    returncode, stdout, stderr = ExecCommand(cmd, environ={'FSLOUTPUTTYPE':'NIFTI'}).run()
    if returncode != 0:
        raise DiciphrException('Error encountered while runnig flirt')
    out_mat = np.loadtxt(outmat_filename).astype(np.float32)
    tmp_im = nib.load(output_filename)
    out_im = nib.Nifti1Image(tmp_im.get_data(), tmp_im.affine, tmp_im.header)
    shutil.rmtree(tmpdir)
    if return_mat and return_im: 
        return (out_im, out_mat)
    elif return_mat:
        return out_mat
    else:
        return out_im
        
def write_flirt_matrix_to_file(flirt_mat, filename):
    np.savetxt(filename, flirt_mat, fmt='%0.12f')
   
def fast_segmentation(nifti_im, type='t1'):
    '''Runs fast segmentation and returns csf, gm, and wm partial volume maps. 
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be masked using fsl bet2
    type : Optional[str]
        't1', 't2', or 'pd' 
    Returns
    -------
    tuple (nibabel.Nifti1Image,nibabel.Nifti1Image,nibabel.Nifti1Image)
        csf, gm, and wm partial volume maps
    '''
    if type.lower() == 't1':
        type_value = '1'
    elif type.lower() == 't2':
        type_value = '2'
    elif type.lower() == 'pd':
        type_value = '3'
    tmpdir = make_temp_dir(prefix='fast_segmentation')
    try:
        tmp_filename = os.path.join(tmpdir,'input.nii.gz')
        tmp_prefix = os.path.join(tmpdir,'fast')
        nifti_im.to_filename(tmp_filename)
        cmd = ['fast', '-t',type_value, '-o',tmp_prefix, tmp_filename]
        ExecCommand(cmd).run()
        csf_filename = find_nifti_from_prefix(tmp_prefix+'_pve_0')
        gm_filename = find_nifti_from_prefix(tmp_prefix+'_pve_1')
        wm_filename = find_nifti_from_prefix(tmp_prefix+'_pve_2')
        csf_result_im = nib.load(csf_filename)
        gm_result_im = nib.load(gm_filename)
        wm_result_im = nib.load(csf_filename)
        #make a new image object to load data into memory so tmpdir can be deleted
        #also removes filename attribute of header which refers to a dir which will be removed
        csf_out_im = nifti_image(csf_result_im.get_data(), csf_result_im.affine, csf_result_im.header)
        gm_out_im = nifti_image(gm_result_im.get_data(), gm_result_im.affine, gm_result_im.header)
        wm_out_im = nifti_image(wm_result_im.get_data(), wm_result_im.affine, wm_result_im.header)
    except Exception as err:
        raise err
    finally:
        shutil.rmtree(tmpdir)
    return csf_out_im, gm_out_im, wm_out_im
    
def bet2_mask_nifti(nifti_im, f=0.2, erode=True):
    '''Mask a nifti image with bet2. Optionally erode the mask by a 2mm sphere
    
    Parameters
    ----------
    nifti_im : nibabel.Nifti1Image
        A nifti image to be masked using fsl bet2
    f : float
        fractional intensity threshold (0->1)
    erode : bool
        If true, result of bet2 will be eroded with
        fslmaths -ero -kernel -sphere 2 
    
    Returns
    -------
    nibabel.Nifti1Image
        The mask image
    '''
    tmpdir = make_temp_dir(prefix='bet2_mask_nifti')
    try:
        f = float(f)
        tmp_filename = os.path.join(tmpdir,'input.nii.gz')
        mask_tmp_betprefix = os.path.join(tmpdir,'input')
        mask_tmp_fileprefix = os.path.join(tmpdir,'input_mask')
        mask_ero_tmp_fileprefix = os.path.join(tmpdir,'input_mask_ero')
        nifti_im.to_filename(tmp_filename)
        cmd = ['bet2',tmp_filename,mask_tmp_betprefix,'-nm','-f',str(f)]
        ExecCommand(cmd).run()
        if erode:
            cmd = ['fslmaths',mask_tmp_fileprefix,'-ero','-kernel','sphere','2',mask_ero_tmp_fileprefix]
            ExecCommand(cmd).run()
            result_filename=find_nifti_from_prefix(mask_ero_tmp_fileprefix)
        else:
            result_filename=find_nifti_from_prefix(mask_tmp_fileprefix)
        result_im = nib.load(result_filename)
        #make a new image object to load data into memory so tmpdir can be deleted
        #also removes filename attribute of header which refers to a dir which will be removed
        out_im = nifti_image(result_im.get_data(),result_im.affine,result_im.header)
    except Exception as err:
        raise err
    finally:
        shutil.rmtree(tmpdir)
    return out_im
    
def extract_roi(atlas_im, roi_index):
    '''
    
    '''
    if not 'int' in str(atlas_im.get_data_dtype()):
        raise DiciphrException('Will not try to extract ROI from non-integer image')
    atlas_data = atlas_im.get_data()
    roi_data = (atlas_data==int(roi_index)).astype(atlas_im.get_data_dtype())
    roi_im = nifti_image(roi_data, atlas_im.affine, atlas_im.header)
    return roi_im
    
def center_of_mass_atlas(atlas_im, labels=[], return_voxel_coordinates=False):
    from collections import OrderedDict
    atlas_data = atlas_im.get_data().astype(np.int16)
    atlas_affine = atlas_im.affine
    if not labels:
        labels = np.unique(atlas_data[atlas_data>0])
    coordinate_coms = OrderedDict()
    for label in labels:
        x, y, z = np.where(atlas_data == label)
        x_com = np.mean(x.astype(np.float32))
        y_com = np.mean(y.astype(np.float32))
        z_com = np.mean(z.astype(np.float32))
        if not return_voxel_coordinates:
            voxel_vector = np.array([[x_com],[y_com],[z_com],[1]])
            coordinate_coms[label] = list(np.dot(atlas_affine, voxel_vector).flatten()[:3])
        else:
            coordinate_coms[label] = [ int(np.round(x_com)), int(np.round(y_com)), int(np.round(z_com)) ]
    return coordinate_coms  
  
def split_image(nifti_im, dimension='t', slice_index=None):
    '''Split an image along a given dimension and save to output_filebase.
    
    Parameters
    ----------
        nifti_im : nibabel.Nifti1Image
            The nifti image 
        dimension : Optional[str]
            The dimension to split, defaults to 't', must be one of ( 't', 'x', 'y', 'z' )
        slice_index : Optional[int]
            If provided, will save only this slice number and not all slices.
    Returns
    -------
        list
            A list of nibabel.Nifti1Image objects
    '''
    _dimensions = ('x','y','z','t')
    if not dimension in _dimensions:
        raise DiciphrException('dimension argument must be one of ( t, x, y, z )')
    if _dimensions.index(dimension) + 1 > len(nifti_im.shape):
        raise DiciphrException('dimension argument out of range for image')
    data = nifti_im.get_data()
    affine = nifti_im.affine
    header = nifti_im.header
    out_images=[]
    if dimension == 'z':
        if slice_index is None:
            for idx in range(nifti_im.shape[2]):
                slice_im = nifti_image(data[:,:,idx,...], affine, header)
                out_images.append(slice_im)
        else:   
            slice_im = nifti_image(data[:,:,slice_index,...], affine, header)
            return slice_im 
    elif dimension == 'y':
        if slice_index is None:
            for idx in range(nifti_im.shape[1]):
                slice_im = nifti_image(data[:,idx,...], affine, header)
                out_images.append(slice_im)
        else:   
            slice_im = nifti_image(data[:,slice_index,...], affine, header)
            return slice_im 
    elif dimension == 'x':
        if slice_index is None:
            for idx in range(nifti_im.shape[0]):
                slice_im = nifti_image(data[idx,...], affine, header)
                slice_im = nifti_image(data[idx,...], affine, header)
                out_images.append(slice_im)
        else:   
            slice_im = nifti_image(data[slice_index,...], affine, header)
            return slice_im
    elif dimension == 't':
        if slice_index is None:
            for idx in range(nifti_im.shape[-1]):
                slice_im = nifti_image(data[...,idx], affine, header)
                out_images.append(slice_im)
        else:   
            slice_im = nifti_image(data[...,slice_index], affine, header)
            return slice_im 
    return out_images
    
def crop_pad_image(nifti_im, x_adjust=[0,0], y_adjust=[0,0], z_adjust=[0,0]):
    '''Crop or pad a Nifti image.
    
    Parameters
    ----------
        nifti_im : nibabel.Nifti1Image
            The nifti image 
        x_adjust : Optional[list]
            A list [ x_adjust_low, x_adjust_high ] of the adjustment to make on either side of the x-axis. To crop, use negative integer, to pad, use positive. 
        y_adjust : Optional[list]
            A list of the adjustment to make on either side of the y-axis. 
        z_adjust : Optional[list]
            A list of the adjustment to make on either side of the z-axis. 
    Returns
    -------
        nib.Nifti1Image
            The output nifti image
    '''  
    if (len(x_adjust) != 2) or (len(y_adjust) != 2) or (len(z_adjust) != 2):
        raise DiciphrException('Cannot parse adjustment inputs')
    for _el in sum((x_adjust, y_adjust, z_adjust), []):
        if not isinstance(_el, int): 
            raise DiciphrException('Adjustment values must be int')
    aff = nifti_im.affine
    hdr = nifti_im.header
    orig_shape = nifti_im.header.get_data_shape()
    ndims = np.array(orig_shape)
    ndims[0] += np.sum(x_adjust)
    ndims[1] += np.sum(y_adjust)
    ndims[2] += np.sum(z_adjust)
    nx, ny, nz = orig_shape[:3]
    new_origin = np.dot(aff,np.array([-1*x_adjust[0],-1*y_adjust[0],-1*z_adjust[0],1]))[0:3]
    x_range_new = [max([0,x_adjust[0]]), x_adjust[0]+orig_shape[0]+min([0,x_adjust[1]])]
    y_range_new = [max([0,y_adjust[0]]), y_adjust[0]+orig_shape[1]+min([0,y_adjust[1]])]
    z_range_new = [max([0,z_adjust[0]]), z_adjust[0]+orig_shape[2]+min([0,z_adjust[1]])]
    x_range_old = [max([0,-1*x_adjust[0]]), max([0,-1*x_adjust[0]]) + x_range_new[1]-x_range_new[0]]
    y_range_old = [max([0,-1*y_adjust[0]]), max([0,-1*y_adjust[0]]) + y_range_new[1]-y_range_new[0]]
    z_range_old = [max([0,-1*z_adjust[0]]), max([0,-1*z_adjust[0]]) + z_range_new[1]-z_range_new[0]]
    orig_data = nifti_im.get_data()
    new_data = np.zeros(ndims,nifti_im.get_data_dtype())
    #copy the data over...
    if len(ndims) == 3:
        new_data[x_range_new[0]:x_range_new[1],y_range_new[0]:y_range_new[1],z_range_new[0]:z_range_new[1]] = orig_data[x_range_old[0]:x_range_old[1],y_range_old[0]:y_range_old[1],z_range_old[0]:z_range_old[1]]
    else:
        new_data[x_range_new[0]:x_range_new[1],y_range_new[0]:y_range_new[1],z_range_new[0]:z_range_new[1],...] = orig_data[x_range_old[0]:x_range_old[1],y_range_old[0]:y_range_old[1],z_range_old[0]:z_range_old[1],...]
    new_aff=aff.copy()
    new_aff[0:3,3]=new_origin
    new_im = nifti_image(new_data,new_aff)
    new_im.header['cal_max'] = hdr['cal_max']
    new_im.header['cal_min'] = hdr['cal_min']
    new_im.update_header()
    return new_im
    
def check_affines_and_shapes_match(*images):
    logging.debug('diciphr.nifti_utils.check_affines_and_shapes_match')
    affines = [im.affine for im in images]
    shapes = [im.shape for im in images]
    _affine_check = np.sum(np.array([np.abs(_a - affines[0]) for _a in affines]))
    if _affine_check > 1e-4:
        logging.debug('Affines: {}'.format(affines))
        raise DiciphrException('Affines do not match!')
    _shape_check = np.sum(np.array(shapes) != np.array(shapes[0])[None, ...])
    if _shape_check > 0:
        logging.debug('Shapes: {}'.format(shapes))
        raise DiciphrException('Shapes do not match!')
    
def multiply_images(*images):
    logging.debug('diciphr.nifti_utils.multiply_images')
    check_affines_and_shapes_match(*images)
    aff = images[0].affine
    result_data = images[0].get_data()
    if len(images) > 1:
        for im in images[1:]:
            result_data = result_data * im.get_data()
    result_im = nifti_image(result_data, aff)
    return result_im
    
def add_images(*images, **kwargs):
    logging.debug('diciphr.nifti_utils.add_images')
    check_affines_and_shapes_match(*images)
    binarize = kwargs.get('binarize', False)
    aff = images[0].affine
    if binarize:
        result_data = (images[0].get_data() > 0).astype(np.int16)
    else:
        result_data = images[0].get_data()
    if len(images) > 1:
        for im in images[1:]:
            if binarize:
                result_data = ((result_data + im.get_data()) > 0).astype(np.int16)
            else:
                result_data = result_data + im.get_data()
    result_im = nifti_image(result_data, aff)
    return result_im
    
def threshold_image(nifti_img, threshold_value):
    data = nifti_img.get_data()
    data_t = (data > threshold_value)*1 
    img = nifti_image(data, nifti_img.affine)
    return img 
    
def replace_labels(atlas_im, input_list, output_list):
    '''
    Replace labels in an atlas.

    Parameters
    ----------
    atlas_im : nibabel.Nifti1Image
        Probtrackx directory.
    input_list : list
        Input list of labels
    output_list : list
        Output list of labels.
    Returns
    -------
    None
    '''
    atlas_data = atlas_im.get_data()
    atlas_affine = atlas_im.affine
    atlas_header = nib.Nifti1Header()
    atlas_header.set_sform(atlas_affine)
    atlas_header.set_qform(atlas_affine)
    logging.debug('Input list: '+', '.join(map(str,input_list)))
    logging.debug('Output list: '+', '.join(map(str,output_list)))
    new_atlas_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for _i, _o in zip(input_list, output_list):
        new_atlas_data[atlas_data == _i] = _o
    new_atlas_im = nib.Nifti1Image(new_atlas_data, atlas_affine, atlas_header)
    return new_atlas_im

def zeros_nifti(nifti_im):
    return nifti_image(np.zeros(nifti_im.shape, dtype=np.uint8), nifti_im.affine)
    
def ones_nifti(nifti_im):
    return nifti_image((nifti_im.get_data()>0).astype(np.uint8), nifti_im.affine)
    
def invert_mask(nifti_im):
    return nifti_image((nifti_im.get_data()==0).astype(np.uint8), nifti_im.affine)
       
def concatenate_niftis(*nifti_ims):
    affine = nifti_ims[0].affine
    data = []
    for img in nifti_ims:
        d = img.get_fdata()
        if len(d.shape) == 3:
            d = d[...,np.newaxis]
        data.append(d)
    dataC = np.concatenate(data, axis=-1)
    return nifti_image(dataC, affine)

def ants_registration_syn(moving_images, fixed_images, output_prefix, mask_images=[], initial_transform_filenames=[], 
                        transform_type='s', dimensionality=3, radius=4, spline_distance=26, 
                        precision='f', histogram_matching=False, collapse_transforms=1):
    fixed_images = force_to_list(fixed_images)
    moving_images = force_to_list(moving_images)
    initial_transform_filenames = force_to_list(initial_transform_filenames)
    mask_images = force_to_list(mask_images)
    
    tmpdir = make_temp_dir(prefix='ants_registration')
    try:
        fixed_filenames = [ os.path.join(tmpdir, 'fixed{}.nii.gz'.format(i)) for i in range(len(fixed_images)) ]
        moving_filenames = [ os.path.join(tmpdir, 'moving{}.nii.gz'.format(i)) for i in range(len(moving_images)) ]
        mask_filenames = [ os.path.join(tmpdir, 'mask{}.nii.gz'.format(i)) for i in range(len(mask_images)) ]
        for im, fn in zip(fixed_images, fixed_filenames):
            write_nifti(fn, im)
        for im, fn in zip(moving_images, moving_filenames):
            write_nifti(fn, im)
        for im, fn in zip(mask_images, mask_filenames):
            write_nifti(fn, im)
        cmd = ['antsRegistrationSyN.sh', 
                '-d', str(dimensionality), 
                '-f', ' '.join(fixed_filenames), 
                '-m', ' '.join(moving_filenames), 
                '-o', output_prefix,
                '-t', transform_type,
                '-r', str(radius), 
                '-s', str(spline_distance), 
                '-p', precision, 
                '-j', '1' if histogram_matching else '0' , 
                '-z', str(collapse_transforms)
        ]
        if mask_images:
            cmd.extend(['-x', ' '.join(mask_filenames)])
        if initial_transform_filenames:
            cmd.extend(['-x', ' '.join(initial_transform_filenames)])
        ExecCommand(cmd).run()        
    except Exception as err:
        raise err
    finally:
        shutil.rmtree(tmpdir)                    
    
def ants_apply_transforms(input_filename, output_filename, reference_filename, transform_filenames, invert=[], interpolation='Linear', bg_value=0):
    '''
    Runs antsApplyTransforms 

    Parameters
    ----------
    input_filename : str
        input nifti image file
    output_filename : str
        output  nifti image file
    reference_filename : str
        reference nifti image file
    transform_filenames : list
        transform files, as a list 
    invert : Optional[list]
        A list of 1s and 0s, that corresponds to transform_filenames: 1 for invert the transform at that position in the list, 0 for not invert. 
    interpolation : Optional[str]
        The interpolation argument, one of 'Linear', 
                        'NearestNeighbor', 
                        'MultiLabel[<sigma=imageSpacing>,<alpha=4.0>]', 
                        'Gaussian[<sigma=imageSpacing>,<alpha=1.0>]'.
                        'BSpline[<order=3>]',
                        'CosineWindowedSinc',
                        'WelchWindowedSinc',
                        'HammingWindowedSinc',
                        'LanczosWindowedSinc',
                        'GenericLabel[<interpolator=Linear>]'
    bg_value : Optional[float]
        The background fill value of the data, usually 0, but sometimes 1 for e.g. FW VF map. 
    Returns
    -------
    None
    '''
    if len(invert) == 0:
        invert = [ 0 for tf in transform_filenames ] 
    transform_string = ','.join( [tf if i==0 else '[{},1]'.format(tf) for tf, i in zip(transform_filenames, invert) ])
    cmd = [ 'antsApplyTransforms', 
                '-i', input_filename, 
                '-o', output_filename, 
                '-r', reference_filename, 
                '-t', transform_string,
                '-n', interpolation, 
                '-f', str(bg_value),
                '-v', '1' 
    ]
    ExecCommand(cmd).run() 
                
def read_ants_affine_transform(transform_file):
    '''
    Call command antsTransformInfo to read an itk transform file.
    '''
    import subprocess
    antsTransformInfo_cmd = which('antsTransformInfo')
    transformInfoOutput = subprocess.check_output([antsTransformInfo_cmd, transform_file], shell=False).decode('ascii')

    matrix = transformInfoOutput.split('Matrix:')[1].split('Offset:')[0]
    matrix = matrix.strip().split('\n')
    matrix = [ a.strip().split() for a in matrix]
    matrix = np.array(matrix).astype(float)

    offset = transformInfoOutput.split('Offset:')[1].split('Center:')[0]
    offset = offset.split('[')[1].split(']')[0].split(',')
    offset = np.array(offset).astype(float)

    center = transformInfoOutput.split('Center:')[1].split('Translation:')[0]
    center = center.split('[')[1].split(']')[0].split(',')
    center = np.array(center).astype(float)

    translation = transformInfoOutput.split('Translation:')[1].split('Inverse:')[0]
    translation = translation.split('[')[1].split(']')[0].split(',')
    translation = np.array(translation).astype(float)

    inverse = np.linalg.inv(matrix)
    
    ret = dict((('Matrix',matrix), 
            ('Offset',offset), 
            ('Center',center), 
            ('Translation',translation), 
            ('Inverse',inverse)))
    return ret
