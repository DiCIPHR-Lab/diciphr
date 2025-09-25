# -*- coding: utf-8 -*-

import os, sys, shutil, logging
import numpy as np
import nibabel as nib

def density(connmat, threshold=0):
    N = connmat.shape[0]
    return (connmat>threshold).sum()/(N*(N-1.))
    
def degree(connmat):
    return np.sum(connmat>0,axis=1)

def nodestrength(connmat):
    return np.sum(connmat,axis=1)
    
def gmwmi_normalize(connmat, gmwmi_im, symmetrize=True, zero_diagonal=True):
    gmwmi_data = gmwmi_im.get_fdata()
    zooms = gmwmi_im.header.get_zooms()
    voxels = ((gmwmi_data > 0)*1).sum()
    volume = voxels*zooms[0]*zooms[1]*zooms[2]
    connmat = connmat / volume 
    if symmetrize:
        connmat = (connmat+connmat.T)/2.0
    if zero_diagonal:
        np.fill_diagonal(connmat, 0)
    return connmat 
    
def volume_normalize(connmat, atlas_im, labels, symmetrize=True, zero_diagonal=True):
    atlas_data=atlas_im.get_fdata()
    volumes = np.array([(atlas_data==l).sum() for l in labels])
    volume_mat = np.array([volumes for a in range(len(volumes))])
    connmat_normalized = connmat / volume_mat
    if symmetrize:
        connmat_normalized = (connmat_normalized+connmat_normalized.T)/2.0
    if zero_diagonal:
        np.fill_diagonal(connmat_normalized, 0)
    return connmat_normalized, volume_mat
    
def read_connmat(f, delimiter=' ', fill_diagonal=0, symmetrize=True):
    mat = np.loadtxt(f, delimiter=delimiter).astype(np.float32)
    if fill_diagonal:
        np.fill_diagonal(mat,fill_diagonal)
    if symmetrize:
        mat = (mat+mat.T)/2.0
    return mat
 
def square_to_ut(mat, diagonal=False):
    return mat[ut_indices(mat, diagonal=diagonal)]
    
def ut_to_square(array, default_value=0, diagonal=False):
    N = len(array)
    nedges = 0.5*(np.sqrt(8*N + 1)+1)
    if not nedges.is_integer():
        raise ValueError('Input array cannot be converted to square matrix')
    nedges = int(nedges)
    if diagonal:
        nedges -= 1 
    ret = np.full((nedges, nedges), default_value)
    ret[ut_indices(ret, diagonal=diagonal)] = array
    ret = ret.transpose()
    ret[ut_indices(ret, diagonal=diagonal)] = array
    return ret
    
def is_binary(connmat):
    _nonzero_elements = np.unique(connmat[connmat!=0])
    if len(_nonzero_elements) == 1:
        return True
    else:
        return False

def binarize_mat(connmat):
    '''Binarize a matrix''' 
    return (connmat>0)*1
    
def normalize_mat(connmat):
    '''Min-max normalize a matrix''' 
    return (connmat.astype(np.float32) - np.min(connmat))/(np.max(connmat) - np.min(connmat))

def fischer_z_transform(connmat):
    ''' Fischer Z transform, e.g. for a functional connectome '''
    if np.min(connmat) < -1 or np.max(connmat) > 1:
        raise DiciphrException('connmat has values outside the range [-1,1].')
    return 0.5 * np.log((1+connmat)/(1-connmat))
    
def ut_indices(mat, diagonal=False):
    if diagonal:
        return np.triu_indices(mat.shape[0],0)
    else:
        return np.triu_indices(mat.shape[0],1)

def lt_indices(mat, diagonal=False):
    if diagonal:
        return np.tril_indices(mat.shape[0],0)
    else:
        return np.tril_indices(mat.shape[0],1)

def prune_mat(connmat, density_target=0.15, abs=False):
    ''' Prune a single connectome to a desired target density, keeping the strongest edges. If abs=True, will return strongest positive or negative edges.'''
    N = connmat.shape[0]
    if density_target > 1:
        density_target = float(density_target)/100
    if density_target == 1:
        return connmat.copy()
    if abs:
        V = sorted(np.abs(connmat)[ut_indices(connmat)], reverse=True)
    else:
        V = sorted(connmat[ut_indices(connmat)], reverse=True)
    nb_keep = int(density_target*len(V))
    threshold_value = V[nb_keep]
    connmat_thresh = connmat.copy()
    if abs:
        connmat_thresh[np.abs(connmat_thresh) < threshold_value] = 0 
    else:        
        connmat_thresh[connmat_thresh < threshold_value] = 0 
    return connmat_thresh
    
def log_scale_mat(connmat, base=10):
    if base == 'e':
        base = np.e
    connmat = np.log(connmat)
    connmat[np.isinf(connmat)] = 0
    connmat = connmat/np.log(base)
    return connmat 
    
def coefficient_of_variation(mat_array):
    means = np.nanmean(mat_array, axis=0)
    stdevs = np.nanstd(mat_array, axis=0, ddof=0)
    cv = stdevs / np.abs(means)
    return cv 
    
def consistency_filter_connmat(mat_array, controls_mat_array, target_density=None, target_cv=0.33333333333333,
        return_mask=False):
    ''' cv_filter_connmat(mat_array) 
    mat_array shape is n_subjects,n_nodes,n_nodes
    controls_mat_array shape is m_subjects, n_nodes, n_nodes
    target_density is an float between 0 and 1 or an int between 1 and 100 
    target_cv is a float.
    '''
    cv = coefficient_of_variation(controls_mat_array)
    if target_density is not None:
        if target_density <= 1: 
            target_density = target_density*100
        target_density = int(target_density)
        target_cv = np.percentile(cv[ut_indices(cv)],int(target_density))
    mask = ( cv < target_cv )
    logging.info("Coeff. of variation (std/mean)={0}, density={1}".format(target_cv, density(mask*1)))
    mat_array = (mat_array * mask[None,...])
    if return_mask:
        mask[lt_indices(mask)] = False
        return (mat_array, mask)
    else:
        return mat_array    
    
def symmetric_mat_from_upper_mask(vector, upper_triangular_mask, fill_value=0.0):
    mat = np.zeros(upper_triangular_mask.shape) + fill_value
    mat[upper_triangular_mask] = vector
    mat = mat.T
    mat[upper_triangular_mask] = vector
    return mat
    
def rand_index(mask):
    edge_indices=np.array(np.where(mask))
    return tuple(edge_indices[:,np.random.randint(edge_indices.shape[1])])
