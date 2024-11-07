# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import pandas as pd
from diciphr.nifti_utils import replace_labels, read_nifti
from scipy.stats import kurtosis, skew

def registration_check_background(scalar_img, labels_img, exception='raise'):
    # Counts how many background (zero) voxels are covered 
    # by labels in registered atlas image 
    try:
        scalar_data = scalar_img.get_fdata()
        labels_data = labels_img.get_fdata()
        zooms = scalar_img.header.get_zooms()
        bgvoxels = ((scalar_data == 0)*(labels_data>0)).sum()
        bgvolume = bgvoxels*zooms[0]*zooms[1]*zooms[2]
        fgvoxels = ((scalar_data != 0)*(labels_data>0)).sum()
        fgvolume = fgvoxels*zooms[0]*zooms[1]*zooms[2]
        try:
            ratio = bgvolume / fgvolume 
        except ZeroDivisionError:
            ratio = np.nan 
        return bgvolume, fgvolume, ratio
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 

def corpus_callosum_distribution(scalar_img, cc_roi_img):
    roi_mask = cc_roi_img.get_fdata()>0
    scalar_data = scalar_img.get_fdata()[roi_mask]
    bin_edges = np.linspace(0,1,21,endpoint=True)
    mean_ = np.mean(scalar_data)
    std_ = np.std(scalar_data)
    kurt_ = kurtosis(scalar_data)
    skew_ = skew(scalar_data)
    histogram, __ = np.histogram(scalar_data, bins=bin_edges, density=True)
    return mean_, std_, skew_, kurt_, histogram, bin_edges
    
def extract_corpus_callosum_from_eve(eve_labels_img):
    input_list = [52,53,54,140,141,142]
    output_list = [1,1,1,1,1,1]
    return replace_labels(eve_labels_img, input_list, output_list)
    
def registration_qc_series(fa_filename, eve_labels_filename, exception='raise'):
    try:
        fa_img = read_nifti(fa_filename)
        eve_labels_img = read_nifti(eve_labels_filename)
        row = pd.Series(name=fa_filename)
        bgvolume, fgvolume, ratio = registration_check_background(fa_img, eve_labels_img)
        row.loc['bg_volume'] = bgvolume
        row.loc['fg_volume'] = fgvolume
        row.loc['bgfg_ratio'] = ratio
        mean_, std_, skew_, kurt_, histogram, bin_edges = corpus_callosum_distribution(fa_img, extract_corpus_callosum_from_eve(eve_labels_img))
        row.loc['mean'] = mean_
        row.loc['std'] = std_
        row.loc['skew'] = skew_
        row.loc['kurt'] = kurt_
        for i, v in enumerate(histogram):
            low_ = bin_edges[i]
            high_ = bin_edges[i+1]
            row.loc['hist{0:0.02f}-{1:0.02f}'.format(low_, high_)] = histogram[i]
        return row 
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 