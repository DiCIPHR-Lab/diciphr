# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import pandas as pd
from diciphr.nifti_utils import read_nifti
from diciphr.diffusion import bet2_mask_nifti

def mias(nifti_img, exception='raise'):
    # count volume (mm2) of non-zero voxels in Most Inferior Axial Slice (MIAS)
    try:
        mask = (nifti_img.get_fdata() > 0)*1
        zooms = nifti_img.header.get_zooms()
        return mask[:,:,0].sum()*zooms[0]*zooms[1]*zooms[2]
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 
            
def msas(nifti_img, exception='raise'):
    # count volume (mm2) of non-zero voxels in Most Superior Axial Slice (MSAS)
    try:
        mask = (nifti_img.get_fdata() > 0)*1
        zooms = nifti_img.header.get_zooms()
        return mask[:,:,-1].sum()*zooms[0]*zooms[1]*zooms[2]
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 
            
def fov_series(b0_filename, mask=True, exception='raise'):
    try:
        b0_img = read_nifti(b0_filename)
        if mask:
            b0_img, mask_img = bet2_mask_nifti(b0_img, return_brain=True)
        mias_ = mias(b0_img)
        msas_ = msas(b0_img)
        row = pd.Series(name=b0_filename)
        row.loc['MIAS_volume'] = mias_
        row.loc['MSAS_volume'] = msas_
        return row 
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 