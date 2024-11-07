# -*- coding: utf-8 -*-

import os, shutil, logging
import numpy as np
import nibabel as nib
import pandas as pd
from ..utils import ( make_dir, make_temp_dir, is_string, 
                ExecCommand, DiciphrException, which )
from ..nifti_utils import read_nifti, write_nifti, strip_nifti_ext
from ..diffusion import TensorScalarCalculator, is_tensor
from collections import OrderedDict
from itertools import product

def scalar_roi_stats(scalar_im, atlas_im, measures=['mean','median','std','volume'], nonzero=True, 
                    labels=None, index=None, roi_names=None, outlier_thresh=6, mask_im=None):
    logging.debug('diciphr.statistics.dti_roi_stats')
    scalar_data = scalar_im.get_fdata()
    atlas_data = atlas_im.get_fdata().astype(int)
    if labels is None:
        labels = range(1,atlas_data.max()+1)
    num_labels = len(labels)
    if mask_im is not None:
        mask = mask_im.get_data() > 0 
    else:
        mask = atlas_data > 0
    if nonzero:
        mask = np.logical_and(mask, scalar_data != 0)        
    scalar_data = scalar_data.astype(np.float32)[mask]
    scalar_means = np.zeros(num_labels,)
    scalar_stds = np.zeros(num_labels,)
    scalar_medians = np.zeros(num_labels,)
    atlas_volumes = np.zeros(num_labels,)
    atlas_data = atlas_data[mask]
    zoom_factor = atlas_im.header.get_zooms()
    zoom_factor = zoom_factor[0]*zoom_factor[1]*zoom_factor[2]
    
    for i,lbl in enumerate(labels):
        atlas_volumes[i] = zoom_factor*np.sum(atlas_data == lbl)
        if atlas_volumes[i] > 0:
            roi_data = scalar_data[atlas_data==lbl]
            _n_data = len(roi_data)
            # to keep track of outliers 
            quartiles = (np.percentile(roi_data, 25),np.percentile(roi_data, 75))
            iqr = 1.5*(quartiles[1]-quartiles[0])
            outlier_low = quartiles[0]-outlier_thresh*iqr
            outlier_high = quartiles[1]+outlier_thresh*iqr
            n_outliers = np.logical_or(roi_data<=outlier_low,roi_data>=outlier_high).sum()
            if n_outliers > 0:
                logging.warning("Encountered {0} outlier voxels ({1:0.1f} %) within region {2} {3} for subject {4}".format(
                    n_outliers, 100*float(n_outliers)/_n_data, lbl, roi_names[i], index))
            if 'mean' in measures:
                scalar_means[i] = np.mean(roi_data)
            if 'median' in measures:
                scalar_medians[i] = np.median(roi_data)
            if 'std' in measures:
                scalar_stds[i] = np.std(roi_data)
        else:
            if index is not None:
                logging.warning("ROI {0} index {1} volume is 0".format(lbl, index))
            else:
                logging.warning("ROI {0} volume is 0".format(lbl))
    ret = {}
    if roi_names is not None:
        columns = roi_names
    else:
        columns = ['ROI{0}'.format(l) for l in labels]
    if index is not None:
        index = [index]
    if 'mean' in measures: 
        ret['mean'] = pd.DataFrame(scalar_means, index=columns, columns=index).transpose()
    if 'median' in measures:
        ret['median'] = pd.DataFrame(scalar_medians, index=columns, columns=index).transpose()
    if 'std' in measures:
        ret['std'] = pd.DataFrame(scalar_stds, index=columns, columns=index).transpose()
    if 'volume' in measures:
        ret['volume'] = pd.DataFrame(atlas_volumes, index=columns, columns=index).transpose()
    return ret

def sample_dti_roistats(datafiles, atlasfiles, measures=['mean','median','std','volume'], 
                    index=None, labels=None, roi_names=None, nonzero=True, outlier_thresh=10, mask_im=None):
    if is_string(atlasfiles):
        # was provided a path not a list of paths 
        atlases = len(datafiles)*[nib.load(atlasfiles)] 
    else:
        atlases = [ nib.load(f) for f in atlasfiles ]
    images = [ nib.load(f) for f in datafiles ]
    kwargs = {'measures':measures, 'labels':labels, 'roi_names':roi_names, 'nonzero':nonzero, 'outlier_thresh':outlier_thresh, 'mask_im':mask_im}
    if index is None:
        index = [None]*len(images)
    results = [scalar_roi_stats(f, a, index=i, **kwargs) for f, a, i in zip(images, atlases, index)]
    # make into dataframes 
    dataframes = {}
    for m in measures:
        dataframes[m] = pd.concat([results[i][m] for i in range(len(images))])
    return dataframes
