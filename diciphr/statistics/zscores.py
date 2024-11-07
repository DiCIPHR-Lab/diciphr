# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import pandas as pd
import statsmodels.formula.api as smf

def corrected_zscores(dataframe, covars=None, columns=None, subset=None, formula=None):
    '''Calculate Z-scores on dataframe. Has option to fit an OLS model to a subset of the data (e.g. Controls), 
    and correct for covariates based on a formula (e.g. Age+Sex), and return z-scores of residuals of that model. 
    For example, one can correct for Age+Sex within controls and calculate z-scores of patients relative to the 
    controls only model. If no subset or formula are given, simple z-scores are calculated. 
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe of features
    covars : Optional[pandas.DataFrame]
        A dataframe with the same idnex as the main dataframe containing covariate data.
    columns : Optional[list]
        A list of column names of features to calculate z-scores. 
    subset : Optional[list]
        A list of index entries over which an OLS model is fit and mean and standard deviation of resiudals are calculated. 
    formula : Optional[str] 
        A patsy-like formula used to correct 
        
    Returns 
    -------
    pandas.DataFrame
        A dataframe of z-scores
    '''
    if columns is None:
        columns = list(dataframe.select_dtypes(include=[np.number]).columns.values)
    if covars is not None:
        dataframe = dataframe.join(covars, how='inner')
    if formula is not None:
        # add a non-numeric character to column names because statsmodels cant interpret names like 001_SPG_L
        rcolumns = ['r'+c for c in columns]
        rename_dict = dict(zip(columns, rcolumns))
        rdataframe = dataframe.rename(rename_dict, axis='columns')
        resids = pd.DataFrame(0.0, columns=rcolumns, index=rdataframe.index)
        for col in rcolumns:
            if subset is None:
                model = smf.ols('{0} ~ {1}'.format(col, formula), data=rdataframe)
                selected = np.array([True for s in rdataframe.index])
            else:
                selected = np.array([s in subset for s in rdataframe.index])
                model = smf.ols('{0} ~ {1}'.format(col, formula), data=rdataframe.loc[selected,:])
            res = model.fit()
            pred = res.predict(rdataframe)
            resids[col] = rdataframe[col] - pred 
        mean_ = np.array(np.mean(resids.loc[selected,:], axis=0))
        std_ = np.array(np.std(resids.loc[selected,:], axis=0))
        zscores = pd.DataFrame((resids.values - mean_[None,:])/std_[None,:], columns=rcolumns, index=dataframe.index)
    else:
        # no correction just z score 
        mean_ = np.array(np.mean(dataframe[columns], axis=0))
        std_ = np.array(np.std(dataframe[columns], axis=0))
        zscores = pd.DataFrame((dataframe[columns].values - mean_[None,:])/std_[None,:], columns=columns, index=dataframe.index)
        rcolumns = columns
    ret = dataframe[columns].copy()
    for c, rc in zip(columns, rcolumns):
        ret.loc[:,c] = zscores.loc[:,rc]
    return ret 

