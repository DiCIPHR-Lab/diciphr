# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import pandas as pd
from diciphr.diffusion import read_dwi, read_nifti, round_bvals

def voxel_dimensions_series(img_filename, exception='raise'):
    try:
        nifti_img = read_nifti(img_filename)
        shape = nifti_img.header.get_data_shape()
        zooms = nifti_img.header.get_zooms()[:3]
        row = pd.Series()
        row.loc['shape'] = shape
        row.loc['zooms'] = zooms
        return row
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 

def voxel_conforms(img_filename, site_series, exception='raise'):
    try:
        site_dims = np.asarray(site_series)
        vox_dims = np.asarray(voxel_dimensions_series(img_filename))
        perc_diff = np.abs(vox_dims - site_dims)/site_dims 
        return perc_diff.max() < 0.001
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 

def bval_report_series(bval_filename, exception='raise'):
    try:
        bvals = round_bvals(np.loadtxt(bval_filename))
        row = pd.Series()
        U = np.unique(bvals)
        for i, b in enumerate(U):
            count = (bvals == b).sum()
            row.loc['bvalue{0}'.format(i+1)] = b
            row.loc['count{0}'.format(i+1)] = count 
        return row 
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 

def bval_conforms(bval_filename, site_series):
    try:
        return (bval_report_series(bval_filename) == site_series).all()
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 

