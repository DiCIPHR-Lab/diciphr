# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import nibabel as nib 
from diciphr.nifti_utils import strip_nifti_ext
from diciphr.qc.registration import registration_qc_series
from diciphr.qc.conformity import voxel_dimensions_series, bval_report_series
from diciphr.qc.conformity import voxel_conforms, bval_conforms
from diciphr.qc.fov import fov_series 
from diciphr.qc.parse_qcnet import qcnet_series 


def input_dataframe(filename):
    df = pd.read_csv(filename, index_col=0, na_values=['.',' ','','#N/A','NA','nan','NaN','NAN'])
    missing_cols = []
    for col in ['Site','Raw_DWI','Processed_T1','Processed_DWI','FA','B0','Eve_Labels','QCNet']:
        if not col in df.columns:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError('Missing columns from dataframe: '+str(missing_cols))
    return df 

def add_series_to_dataframe(df, index, prefix, series, exception='raise'):
    try:
        for col in series.index:
            df.loc[index, prefix+'_'+col] = str(series[col])
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 
        
def qc_dataframe(input_df, site_df=None):
    subjects = input_df.index 
    results_df = pd.DataFrame(index=subjects)
    for s in subjects:
        print('+++', s)
        raw_dwi_filename = input_df.loc[s, 'Raw_DWI']
        site = input_df.loc[s, 'Site']
        results_df.loc[s, 'c0_Site'] = site
        results_df.loc[s, 'c0_Raw_DWI'] = raw_dwi_filename
        raw_bval_filename = strip_nifti_ext(raw_dwi_filename)+'.bval'
        raw_bvec_filename = strip_nifti_ext(raw_dwi_filename)+'.bvec'
        proc_t1_filename = input_df.loc[s, 'Processed_T1']
        proc_dwi_filename = input_df.loc[s, 'Processed_DWI']
        fa_filename = input_df.loc[s, 'FA']
        b0_filename = input_df.loc[s, 'B0']
        eve_labels_filename = input_df.loc[s, 'Eve_Labels']
        qcnet_filename = input_df.loc[s, 'QCNet']
            
        voxel_dimensions_series_dwi_ = voxel_dimensions_series(raw_dwi_filename, exception='pass')
        add_series_to_dataframe(results_df, s, 'c1_DWIVoxels', voxel_dimensions_series_dwi_, exception='pass')
        voxel_dimensions_series_t1_ = voxel_dimensions_series(proc_t1_filename, exception='pass')
        add_series_to_dataframe(results_df, s, 'c2_T1Voxels', voxel_dimensions_series_t1_, exception='pass') 
        bval_report_series_ = bval_report_series(raw_bval_filename, exception='pass')
        add_series_to_dataframe(results_df, s, 'c3_Bvals', bval_report_series_, exception='pass')
        fov_series_ = fov_series(b0_filename, mask=False, exception='pass')
        add_series_to_dataframe(results_df, s, 'c4_FOV', fov_series_, exception='pass')
        qcnet_series_ = qcnet_series(qcnet_filename, raw_dwi_filename, bval_filename=raw_bval_filename, bvec_filename=raw_bvec_filename, threshold=0.5, exception='pass')
        add_series_to_dataframe(results_df, s, 'c5_3dQCnet', qcnet_series_, exception='pass')
        registration_qc_series_ = registration_qc_series(fa_filename, eve_labels_filename, exception='pass')
        add_series_to_dataframe(results_df, s, 'c6_Registration', registration_qc_series_, exception='pass')
        
    columns = list(results_df.columns)
    sorted_columns = sorted(columns, key=lambda c: c.split('_')[0])
    results_df = results_df[sorted_columns]
    columns = results_df.columns
    columns = [c[3:] for c in columns]
    results_df.columns = columns 
    
    return results_df 