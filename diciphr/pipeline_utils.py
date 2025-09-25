# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, sys, logging
import numpy as np
import nibabel as nib
from diciphr.utils import ( DiciphrException, ExecCommand, TempDirManager, is_nifti_file )
from diciphr.nifti_utils import ( nifti_image, read_nifti, find_nifti_from_basename, 
                                 extract_roi, reorient_nifti, dilate_image, erode_image )

def _check_bedpostx_mask_alignment(bedpostx_subj_dir, log_file=None):
    '''Try to find slices in bedpostX output that do not align with the input mask.'''
    logging.debug('diciphr.connectivity._check_bedpostx_mask_alignment')
    merged_filename = os.path.join(bedpostx_subj_dir, 'bedpost.bedpostX', 'merged_f1samples')
    merged_filename = find_nifti_from_basename(os.path.realpath(merged_filename))
    merged_data = read_nifti(merged_filename).get_fdata()
    mask_filename = os.path.join(bedpostx_subj_dir, 'bedpost', 'nodif_brain_mask')
    mask_filename = find_nifti_from_basename(os.path.realpath(mask_filename))
    mask_data = read_nifti(mask_filename).get_fdata()
    nz = mask_data.shape[2]
    ret=''
    for z_idx in range(nz):
        mask_slice = (mask_data[...,z_idx] > 0).astype(np.int16)
        merged_slice = (merged_data[...,z_idx, 0] > 0).astype(np.int16)
        num_mask_voxels = np.sum(mask_slice)
        overlap = np.sum(mask_slice * merged_slice)
        if not overlap == num_mask_voxels:
            message = '{0} slice {1} ... BAD\n'.format(merged_filename, z_idx)
            if log_file:
                log = open(log_file,'a')
                log.write(message)
                log.close()
            ret+=message
            sys.stdout.write(message)
    return ret
    
def _check_bedpostx_niftis(bedpostx_subj_dir, log_file=None):
    '''Perform existence and Nifti quality check on all files in bedpostX output directory.'''
    logging.debug('diciphr.connectivity._check_bedpostx_niftis')
    directory = os.path.join(bedpostx_subj_dir, 'bedpost.bedpostX')
    filenames = ['dyads1_dispersion.nii.gz', 'dyads1.nii.gz', 'dyads2_dispersion.nii.gz', 'dyads2.nii.gz',
        'mean_dsamples.nii.gz', 'mean_f1samples.nii.gz', 'mean_f2samples.nii.gz','mean_ph1samples.nii.gz',
        'mean_ph2samples.nii.gz', 'mean_th1samples.nii.gz','mean_th2samples.nii.gz',
        'merged_f1samples.nii.gz', 'merged_f2samples.nii.gz', 'merged_ph1samples.nii.gz',
        'merged_ph2samples.nii.gz', 'merged_th1samples.nii.gz', 'merged_th2samples.nii.gz',
        'nodif_brain_mask.nii.gz']
    ret=''
    for f in filenames:
        message = ''
        if not os.path.exists(os.path.join(directory, f)):   
            message = 'File missing: {}\n'.format(f)
        else:
            if not is_nifti_file(os.path.join(directory, f)):
                message = 'Nifti error: {}\n'.format(f)
        if message:
            if log_file:
                log = open(log_file,'a')
                log.write(message)
                log.close()
            ret+=message
            sys.stdout.write(message)   
    return ret
    
def check_bedpostx_output(bedpostx_subj_dir, log_file=None):
    '''Check all niftis in bedpostX output and raise a DiciphrException.'''
    logging.debug('diciphr.connectivity.check_bedpostx_output')
    niftis_missing = bool(_check_bedpostx_niftis(bedpostx_subj_dir, log_file=log_file))
    mask_alignment_problem = bool(_check_bedpostx_mask_alignment(bedpostx_subj_dir, log_file=log_file))
    if niftis_missing or mask_alignment_problem:
        raise DiciphrException('BedpostX check failed')

def check_target_labels(target_mask_im,atlas_labels):
    '''If target_mask_im does not contain only the roi labels in list atlas_labels, raise a DiciphrException.'''
    logging.debug('diciphr.connectivity.check_target_labels')
    target_mask_uniq = np.unique(target_mask_im.get_fdata())
    labels_uniq = np.unique([0]+atlas_labels)
    if not (target_mask_uniq == labels_uniq).all():
        raise DiciphrException('Target image labels do not match provided labels')
    
def convert_freesurfer_output_to_nifti_reference_t1(freesurfer_filename, reference_im):
    '''Convert the freesurfer output from mgz to nii.gz, and match the geometry of nifti reference_im, which should be the T1.
    
    Parameters
    ----------
    freesurfer_filename : str
        The .mgz freesurfer result
    reference_im : nibabel.Nifti1Image
        A nifti image
    
    Returns
    -------
    nibabel.Nifti1Image    
    '''
    logging.debug('diciphr.connectivity.convert_freesurfer_output_to_nifti')
    if not os.path.exists(freesurfer_filename):
        raise DiciphrException('File does not exist {}'.format(freesurfer_filename))
    with TempDirManager(prefix='fs2nifti') as manager:
        tmpdir = manager.path()
        fs_nifti = os.path.join(tmpdir,'freesurfer_as_nifti.nii.gz')
        cmd=['mri_convert',freesurfer_filename,fs_nifti]
        ExecCommand(cmd).run()
        fs_im = read_nifti(fs_nifti)
        ref_affine = reference_im.affine
        ref_data = reference_im.get_fdata()
        orientation = ''.join((_.upper() for _ in nib.orientations.aff2axcodes(ref_affine)))
        reoriented_im = reorient_nifti(fs_im, orientation)
        reoriented_data = reoriented_im.get_fdata()
        reoriented_affine = reoriented_im.affine
        reoriented_affine_inv = np.linalg.inv(reoriented_affine)
        new_data = np.zeros(ref_data.shape, dtype=reoriented_im.header.get_data_dtype())
        _already_visited_this_voxel_data = np.zeros(reoriented_data.shape, dtype=np.uint8)
        for ref_ix, ref_iy, ref_iz in zip(*np.nonzero(ref_data)):
            coords = np.dot(ref_affine, np.array([[ref_ix],[ref_iy],[ref_iz],[1]]))
            reoriented_space_voxel_coords = np.dot(reoriented_affine_inv, coords)
            ix = int(reoriented_space_voxel_coords[0][0])
            iy = int(reoriented_space_voxel_coords[1][0])
            iz = int(reoriented_space_voxel_coords[2][0])
            new_data[ref_ix, ref_iy, ref_iz] = reoriented_data[ix, iy, iz]
            _already_visited_this_voxel_data[ix, ix, ix] += 1
        if np.max(_already_visited_this_voxel_data) > 1:
            logging.warning("Voxels were assigned values more than once: {}".format(np.where(_already_visited_this_voxel_data)))
        im_out = nib.Nifti1Image(new_data, ref_affine)
    return im_out
    
def convert_freesurfer_to_nifti(freesurfer_filename, orn_string='LPS'):
    '''Convert the freesurfer output from mgz to nii.gz, and match the geometry of nifti reference_im, which should be the T1.
    
    Parameters
    ----------
    freesurfer_filename : str
        The .mgz freesurfer result
    orn_string : str
        The orientation of the resultant Nifti file. Default LPS
    
    Returns
    -------
    nibabel.Nifti1Image    
    '''
    logging.debug('diciphr.connectivity.convert_freesurfer_to_nifti')
    if not os.path.exists(freesurfer_filename):
        raise DiciphrException('File does not exist {}'.format(freesurfer_filename))
    with TempDirManager(prefix='fs2nifti') as manager:
        tmpdir = manager.path()
        fs_nifti = os.path.join(tmpdir,'nifti_lia.nii.gz')
        cmd=['mri_convert',freesurfer_filename,fs_nifti]
        ExecCommand(cmd).run()
        lia_im = read_nifti(fs_nifti)
        nifti_im = reorient_nifti(lia_im, orientation=orn_string)
    return nifti_im 

labels_86 = [
    1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 
    1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 8, 10, 11, 12, 13, 17, 18, 26, 28, 
    2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 47, 49, 50, 51, 52, 53, 54, 58, 60
]

labels_wm = [2,7,41,46,250,251,252,253,254,255]

labels_87 = [
    1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 
    1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28, 
    2001, 2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 
    2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 47, 49, 50, 51, 52, 53, 54, 58, 60
]

def extract_gm_mask(aparc_aseg_nifti_im, include_brain_stem=False):
    '''Create the GM mask nifti image from Freesurfer output.'''
    logging.debug('diciphr.connectivity.extract_gm_mask')
    free_data = aparc_aseg_nifti_im.get_fdata()
    gm_data = np.zeros(free_data.shape, dtype=np.int16)
    if include_brain_stem:
        labels = labels_87
    else:
        labels = labels_86
    for label in labels:
        gm_data[free_data == label] = 1
    gm_im = nifti_image(gm_data, aparc_aseg_nifti_im.affine)
    gm_labels_im = nifti_image(gm_data*free_data, aparc_aseg_nifti_im.affine)
    check_target_labels(gm_labels_im, labels)
    return gm_im    
    
def extract_wm_mask(aparc_aseg_nifti_im):
    '''Create the WM mask nifti image from Freesurfer output.'''
    logging.debug('diciphr.connectivity.extract_wm_mask')
    free_data = aparc_aseg_nifti_im.get_fdata()
    wm_data = np.zeros(free_data.shape, dtype=np.int16)
    for label in labels_wm:
        wm_data[free_data == label] = 1
    wm_im = nifti_image(wm_data, aparc_aseg_nifti_im.affine)
    return wm_im
    
def wm_gm_boundary(wm_im, gm_im):
    '''Dilate the WM mask with a 1.5 mm Gaussian kernel (fslmaths) 
    and intersect with the GM mask.'''
    logging.debug('diciphr.connectivity.wm_gm_boundary')
    wm_dil_im = dilate_image(wm_im, kernel='gauss', kernel_size=1.5, iterations=1, binarize=False)
    wm_dil_data = wm_dil_im.get_fdata()
    gm_data = gm_im.get_fdata()
    wm_gm_boundary_data = (wm_dil_data * gm_data).astype(np.int16)
    wm_gm_boundary_im = nib.Nifti1Image(wm_gm_boundary_data, gm_im.affine)
    return wm_gm_boundary_im
    
def wm_gm_boundary_add_subctx(wm_gm_boundary_im, gm_labels_im):
    logging.debug('diciphr.connectivity.wm_gm_boundary_add_subctx')
    gm_labels_data = gm_labels_im.get_fdata()
    subctx_labels = list(np.unique(gm_labels_data[gm_labels_data > 0]))
    subctx_labels = list(filter(lambda _x: _x < 100, subctx_labels))
    subctx_labels = list(filter(lambda _x: _x != 47, subctx_labels))
    subctx_labels = list(filter(lambda _x: _x != 8, subctx_labels))
    subctx_boundary_data = np.zeros(gm_labels_data.shape, dtype=gm_labels_data.dtype)
    for label in subctx_labels:
        logging.debug('Eroding subcortical mask label {}'.format(label))
        subctx_label_im = extract_roi(gm_labels_im, label)
        subctx_label_ero_im = erode_image(subctx_label_im, kernel='gauss', kernel_size=1.5, iterations=1, binarize=True)
        subctx_boundary_data += subctx_label_im.get_fdata() - subctx_label_ero_im.get_fdata()
    wm_gm_boundary_data = wm_gm_boundary_im.get_fdata()
    wm_gm_boundary_data = subctx_boundary_data + wm_gm_boundary_data
    wm_gm_boundary_data = (wm_gm_boundary_data > 0).astype(wm_gm_boundary_data.dtype)
    wm_gm_boundary_out_im = nifti_image(wm_gm_boundary_data, wm_gm_boundary_im.affine, wm_gm_boundary_im.header)
    return wm_gm_boundary_out_im

def write_target_masks(target_mask_im,targets_dir):
    '''Write a nifti file in targets_dir for each ROI in the atlas nifti image.'''
    logging.debug('diciphr.connectivity.write_target_masks')
    target_mask_data = target_mask_im.get_fdata().astype(np.int16)
    targets_dir=os.path.realpath(targets_dir)
    with open(os.path.join(targets_dir,'masks.txt'),'w') as targets_text:
        labels = np.unique(target_mask_data[target_mask_data>0])
        for label in labels:
            filename=os.path.join(targets_dir,'target_{}.nii.gz'.format(label))
            data = np.zeros(target_mask_data.shape, dtype=target_mask_data.dtype)
            data[target_mask_data == label] = 1
            im = nifti_image(data, target_mask_im.affine, target_mask_im.header, np.int16)
            im.to_filename(filename)
            targets_text.write(filename+'\n')    

def split_seed_mask(seed_mask_im,seeds_dir,num_masks):
    logging.debug('diciphr.connectivity.split_seed_mask')
    '''Split a probtrackx seed_mask_im into num_masks smaller pieces to run in parallel.'''
    seed_mask_data = (seed_mask_im.get_fdata() > 0 ).astype(np.int32)
    num_seed_voxels = np.sum(seed_mask_data)
    nb_voxels_per_seed = int( num_seed_voxels/num_masks )
    seed_mask_shape = seed_mask_data.shape
    seed_mask_data = seed_mask_data.flatten()
    for i in range(num_masks):
        seedfile = os.path.join(seeds_dir,'seed_{}.nii.gz'.format(i))
        seed_data = np.zeros(seed_mask_data.shape)
        seed_data_ones = np.zeros((num_seed_voxels,))
        start_ind = i*nb_voxels_per_seed
        if i == num_masks-1:
            end_ind = num_seed_voxels
        else:
            end_ind = (i+1)*nb_voxels_per_seed
        seed_data_ones[start_ind:end_ind] = 1
        seed_data[seed_mask_data>0] = seed_data_ones
        seed_data = seed_data.reshape(seed_mask_shape)
        seed_im = nifti_image(seed_data,seed_mask_im.affine,dtype=np.int16)
        seed_im.to_filename(seedfile)
