# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import pandas as pd
from diciphr.diffusion import read_dwi, round_bvals

def qcnet_series(qcnet_csvfile, dwi_filename, bval_filename=None, bvec_filename=None, threshold=0.5, exception='raise'):
    try:
        qcnet_df = pd.read_csv(qcnet_csvfile)
        # "FilePath","Volume","Predicted Class","Probability"
        qcnet_df = qcnet_df.loc[qcnet_df["FilePath"] == dwi_filename,['Volume','Predicted Class','Probability']]
        qcnet_df['Predicted Class'] = ['bad' if p>=threshold else 'good' for p in qcnet_df['Probability']]
        dwi_img, bvals, bvecs = read_dwi(dwi_filename, bval_filename, bvec_filename)
        bvals = round_bvals(bvals).astype(np.int32).flatten()
        U = np.unique(bvals)
        series = pd.Series(name=dwi_filename, dtype=object)
        for i, b in enumerate(U):
            series.loc['bvalue_{}'.format(i)] = b 
            inds = np.where(bvals == b)[0]
            subdf = qcnet_df.loc[[v in inds for v in qcnet_df['Volume']], :]
            nbad = (subdf['Predicted Class'] == 'bad').sum()
            ntot = len(subdf)
            percbad = 100*nbad/float(ntot)
            series.loc['percent_bad_{}'.format(i)] = percbad 
            if b < 50:
                if percbad >= 50:
                    series.loc['qcnet_grade_{}'.format(i)] = 'bad'
                else:
                    series.loc['qcnet_grade_{}'.format(i)] = 'good'
            elif b >= 50:
                if percbad >= 15:
                    series.loc['qcnet_grade_{}'.format(i)] = 'bad'
                else:
                    series.loc['qcnet_grade_{}'.format(i)] = 'good'
        return series 
    except Exception as e:
        if exception == 'raise':
            raise e
        else:
            pass 
    
