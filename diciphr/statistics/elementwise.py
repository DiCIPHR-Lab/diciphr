#! /usr/bin/env python

import sys, traceback, logging, re 
import pandas as pd
import numpy as np
from collections import OrderedDict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import mannwhitneyu, anderson, pearsonr, spearmanr
from ..nifti_utils import nifti_image, intersection_mask
from .stats_utils import fdr, filter_cohort
from ..utils import DiciphrException, check_inputs, make_temp_dir, logical_or, logical_and
from ..connectivity.connmat_utils import ut_to_square
    
def afni_regression(nifti_filename_template, cohort, full_model, columns=None, 
                        reduced_model=None, mask=None, filters=[], 
                        centralize=[], log_scale=False):
    # Refine cohortfile
    for expr in filters:
        cohort, data = filter_cohort(cohort, expr, data=data)
    for key in centralize:
        cohort[key] -= np.mean(cohort[key])
    subjects = list(cohort.index)
    nifti_files = [ nifti_filename_template.format(s=s) for s in subjects ]
    check_inputs(*nifti_files, nifti=True)
    if not mask:    
        mask = intersection_mask(nifti_files)
    N = len(nifti_files)
    # Refine formulas
    formula_full = '~ {}'.format(full_model.replace(':','*'))
    endogs_full = full_model.replace('(','').replace(')','').replace(':','+').replace('*','+').split('+')
    if columns is None:
        columns = endogs_full
    cohort = cohort[endogs_full]
    design_full = dmatrix(formula_full, cohort, return_type="dataframe")
    if reduced_model: 
        formula_reduced = '~ {}'.format(reduced_model)
        design_reduced = dmatrix(formula_reduced, cohort, return_type="dataframe")
    tmpdir = make_temp_dir(prefix="afni_regression")
    full_cmd = ["3dRegAna"]

def assign_to_results(results, key, value, i, M, default=0.0):
    ''' Convenience function to assign a value to an index of a dictionary of arrays if it has the key, or create the array at the key if it does not. ''' 
    if np.isnan(value):
        value = default 
    if key in results.keys():
        results[key][i] = value
    else:
        results[key] = np.zeros((M,)) + default
        results[key][i] = value
            
def elementwise_ols(data, cohort, full_model, reduced_model=None, columns=None, filters=[], centralize=[], 
                        treatments={}, residualize=[], log_scale=False, alpha=0.05):                
    '''Perform an OLS regression on each element of a rectangular array.
    
    Parameters 
    ----------
    data : numpy.ndarray 
        An ndarray of shape (n, m) where n is the number of subjects and m is the number of elements
    cohort : pandas.DataFrame
        A dataframe of shape (n, k) where n is the number of subjects and k is any number of columns
    full_model : str
        A patsy compatible string for the OLS model, such as Group+Age+Sex
    reduced_model : Optional[str]
        If provided, a patsy compatible string for the reduced OLS model, such as Age+Sex. Will perform an F-test against the full model.
    columns : Optional[list]
        A list of variables to save desired t-stat and effect size results, if not provided, will save for each term in "full_model"
    filters : Optional[list] 
        A list of strings to form filters on the cohort and data, such as "Sex=Male", "Age>18" etc. 
    centralize : Optional[list]
        A list of variables to centralize before OLS is performed, such as "Age"
    treatments : Optional[dict]
        A dictionary of variables used to identify the 'baseline' level for categorical variables, e.g. {'Group':'CON'}
    residualize : Optional[list] 
        A list of variables to preserve effects when residualizing data. e.g. ['Age','Sex'] to preserve Group effects and regress out age and sex for subsequent analysis with full_model 'Group+Age+Sex'
    log_scale : Optional[bool]
        If True, take the natural log of the data before analyzing. Default False.
    alpha : Optional[float] 
        The alpha value for confidence intervals and hypothesis testing. Default is 0.05. One-sided tests are not supported but alpha can be set to 0.10 and the results that do not reject your null hypothesis can be removed by the user later. 
    Returns
    -------
    results
        An OrderedDict of elementwise results. 
    residualized_data
        A numpy.ndarray of shape (n, m) containing residualized data, where n is the number of subjects *after filtering* and m is the number of elements. Returned if residualize is provided.
    models_dict
        A dictionary containing the fit OLS models for each element, returned if return_models is True. 
    '''
    logging.debug('diciphr.statistics.elementwise.elementwise_ols')
    if cohort.shape[0] != data.shape[0]:
        raise DiciphrException('Cohort shape does not match data shape in first dimension')
    M = data.shape[1]
    try:
        # convert a pandas DataFrame to numpy ndarray 
        column_names = data.columns
        index = data.index
        data = data.get_values()
    except Exception as e:
        column_names = np.arange(1,M+1)
        index = None
    endogs_full = list(filter(lambda c: bool(c), re.split('\)|\(|\+|\*|\:|\ ', full_model)))
    endogs_full = np.unique([ e.strip() for e in endogs_full ])
    if columns is None:
        columns = endogs_full
    # Refine cohortfile
    for expr in filters:
        logging.debug('Filter cohort with expr {}'.format(expr))
        cohort, data = filter_cohort(cohort, expr, data=data)
    for key in centralize:
        logging.debug('Centralize column {}'.format(key))
        cohort[key] -= np.mean(cohort[key])
    cohort = cohort[endogs_full]
    # Refine formulas
    if treatments:
        for key in treatments:
            logging.debug('Implementing treatment {} with baseline level {}'.format(key, treatments[key]))
            full_model = full_model.replace(key, "C("+key+", Treatment('"+treatments[key]+"'))")
            if reduced_model:
                reduced_model = reduced_model.replace(key, "C("+key+", Treatment('"+treatments[key]+"'))")
    design_full = dmatrix(full_model, cohort, return_type="dataframe")
    if reduced_model:
        design_reduced = dmatrix(reduced_model, cohort, return_type="dataframe")
    if treatments:
        _cols = list(design_full.columns)
        for i, c in enumerate(design_full.columns):
            if c.startswith('C('):
                k = c.split('(')[1].split(',')[0]
                v = c.split(')')[2]
                _cols[i] = k+v
        design_full.columns = _cols 
        if reduced_model:   
            _cols = list(design_reduced.columns)
            for i, c in enumerate(design_reduced.columns):
                if c.startswith('C('):
                    k = c.split('(')[1].split(',')[0]
                    v = c.split(')')[2]
                    _cols[i] = k+v
            design_reduced.columns = _cols    
    resid_exogs = design_full.copy()
    if residualize:
        logging.debug('Will generate residualized data controlling for {}'.format(','.join(map(str,residualize))))
        resid_exogs = pd.DataFrame(index=design_full.index)
        for c in design_full.columns:
            if c == 'Intercept':
                resid_exogs[c] = design_full[c]
            elif any( [ r in c for r in residualize ] ):
                resid_exogs[c] = design_full[c]
            else:
                resid_exogs[c] = 0.0 
    residualized_data = np.zeros(data.shape)
    if log_scale: 
        data = np.log(data)
    
    #Initialize ...
    results = OrderedDict()
    ret_models = {'full':[]}
    if reduced_model:
        ret_models['reduced'] = []
    for i, colname in enumerate(column_names):
        logging.debug('Performing OLS in element {} out of {}'.format(i+1, M))
        Y = data[:,i]
        if np.sum(np.isnan(Y)) > 0:
            nan_i = np.where(np.isnan(Y))[0]
            logging.warning('NaNs encountered in element {}. Will DROP {} NaN values from the OLS model'.format(colname,np.sum(np.isnan(Y))))
            if index is not None:
                logging.warning('Index with NaNs in element {}: {}'.format(colname, index[nan_i]))
        try:
            model_full = sm.OLS(Y, design_full, missing='drop')
            result_full = model_full.fit()
            ret_models['full'].append(result_full)
            f_comp = None
            if reduced_model:
                model_reduced = sm.OLS(Y, design_reduced)
                result_reduced = model_reduced.fit()
                ret_models['reduced'].append(result_reduced)
                f_comp, f_p, __ = result_full.compare_f_test(result_reduced)
            assign_to_results(results,'f_fullvsnull',result_full.fvalue,i,M,0.0)
            assign_to_results(results,'p_fullvsnull',result_full.f_pvalue,i,M,1.0)
            assign_to_results(results,'q_fullvsnull',1.0,i,M,1.0)
            assign_to_results(results,'R2',result_full.rsquared,i,M,0.0)
            assign_to_results(results,'R2_adj',result_full.rsquared_adj,i,M,0.0)
            assign_to_results(results,'df_resid',result_full.df_resid,i,M,0.0)
            assign_to_results(results,'df_model',result_full.df_model,i,M,0.0)
            CI = result_full.conf_int(alpha=alpha)
            for col in design_full.columns:
                for key in filter(lambda x: col in x, result_full.tvalues.keys()):
                    t = result_full.tvalues[key]
                    p = result_full.pvalues[key]
                    coeff = result_full.params[key]
                    beta = coeff*design_full[col].std()/model_full.endog.std()
                    ciwidth = (CI.loc[key,1]-CI.loc[key,0])/2
                    d = 2*t/np.sqrt(result_full.df_resid)
                    assign_to_results(results,'coeff_'+key,coeff,i,M,0.0)
                    assign_to_results(results,'ci-width_'+key,ciwidth,i,M,0.0)
                    assign_to_results(results,'b_'+key,beta,i,M,0.0)
                    assign_to_results(results,'t_'+key,t,i,M,0.0)
                    assign_to_results(results,'d_'+key,d,i,M,0.0)
                    assign_to_results(results,'p_'+key,p,i,M,1.0)
                    assign_to_results(results,'q_'+key,1.0,i,M,0.0)
                    assign_to_results(results,'trendP_{}'.format(key),1 if p<=alpha else 0,i,M,0.0)
                    assign_to_results(results,'fdrQ_{}'.format(key),0.0,i,M,0.0)
            if f_comp is not None:
                assign_to_results(results,'f_fullvsreduced',f_comp,i,M,0.0)
                assign_to_results(results,'p_fullvsreduced',f_p,i,M,1.0)
                assign_to_results(results,'q_fullvsreduced',1.0,i,M,1.0)    
            predicted = result_full.predict(resid_exogs)
            residualized_data[:,i] = data[:,i] - predicted 
        except:
            logging.warning(" - Could not fit model at element {}".format(i))
            logging.warning("".join(traceback.format_exception(*sys.exc_info())))
    # FDR Correction
    pval_keys = list(filter(lambda x: x.startswith('p_'), results.keys() ))
    logging.debug('FDR Correction')
    for pval_key in pval_keys:
        qval_key = 'q_{}'.format(pval_key[2:])
        dval_key = 'd_{}'.format(pval_key[2:])
        tval_key = 't_{}'.format(pval_key[2:])
        fdrQ_key = 'fdrQ_{}'.format(pval_key[2:])
        results[qval_key] = fdr(results[pval_key])
        results[fdrQ_key] = (results[qval_key] <= alpha)*1
    residualized_data = pd.DataFrame(residualized_data, index=cohort.index, columns=column_names)
    return results, ret_models, residualized_data
    
def elementwise_mannwhitneyu(groupA_data, groupB_data):                
    '''Perform the Mann-Whitney U test on each element of a rectangular array.
    
    Parameters 
    ----------
    groupA_data : numpy.ndarray 
        A ndarray of shape (n0, m) where n0 is the number of subjects in groupA and m is the number of elements. 
    groupB_data : numpy.ndarray 
        A ndarray of shape (n1, m) where n1 is the number of subjects in groupB and m is the number of elements. 
        This is the baseline group, positive effect size means group0 is higher and negative means group1 is higher. 
    Returns
    -------
    results
        An OrderedDict of elementwise results. 
    '''
    logging.debug('diciphr.statistics.elementwise.elementwise_mannwhitneyu')
    M = groupA_data.shape[1]
    logging.debug('groupA_data.shape: {}'.format(groupA_data.shape))
    logging.debug('groupB_data.shape: {}'.format(groupB_data.shape))
    #Initialize ...
    results = OrderedDict()
            
    for i in range(M):
        # logging.debug('Performing Mann-Whitney U in element {} out of {}'.format(i+1, M))
        if i % 10000 == 0:
            logging.info('Performing Mann-Whitney U in element {0} out of {1}  [ {2:0.1f}% ] '.format(i+1, M, 100*float(i+1)/M))
        gA = groupA_data[:,i]
        gB = groupB_data[:,i]
        if np.sum(np.isnan(gA)) + np.sum(np.isnan(gB)) > 0:
            logging.warning('NaNs encountered in element {}. Will DROP {} NaN values from the Mann Whitney U calculation'.format(i+1,np.sum(np.isnan(gA)) + np.sum(np.isnan(gB))))
            gA = gA[np.logical_not(np.isnan(gA))]
            gB = gB[np.logical_not(np.isnan(gB))]
        nA = len(gA)
        nB = len(gB)
        try: 
            U, pval = mannwhitneyu(gB, gA, alternative='two-sided')
        except ValueError:
            logging.debug('Mann-Whitney U failed in element {}'.format(i+1))
            U = 0.0 
            pval = 1.0 
        effect_size = 1 - 2*float(U)/(nA * nB)
        assign_to_results(results,'U',U,i,M,0.0)
        assign_to_results(results,'rank-biserial_corr',effect_size,i,M,0.0)
        assign_to_results(results,'pval',pval,i,M,1.0)
        assign_to_results(results,'qval',1.0,i,M,1.0)
        assign_to_results(results,'nA',nA,i,M,0.0)
        assign_to_results(results,'nB',nB,i,M,0.0)
    # FDR Correction
    results['qval'] = fdr(results['pval'])
    trendP_result = results['rank-biserial_corr'].copy()
    trendP_result[results['pval'] > 0.05] = 0.0 
    results['rank-biserial_corr_trendP'] = trendP_result
    fdrQ_result = results['rank-biserial_corr'].copy()
    fdrQ_result[results['qval'] > 0.05] = 0.0  
    results['rank-biserial_corr_fdrQ'] = fdrQ_result
    return results 

def elementwise_anderson_darling(data):  
    '''Perform the Anderson Darling test of normality on each element of a rectangular array.
    
    Parameters 
    ----------
    groupA_data : numpy.ndarray 
        A ndarray of shape (n0, m) where n0 is the number of subjects in groupA and m is the number of elements. 
    groupB_data : numpy.ndarray 
        A ndarray of shape (n1, m) where n1 is the number of subjects in groupB and m is the number of elements. 
        This is the baseline group, positive effect size means group0 is higher and negative means group1 is higher. 
    Returns
    -------
    results
        An OrderedDict of elementwise results. 
    '''
    M = data.shape[1]
    #Initialize ...
    results = OrderedDict()
    for i in range(M):
        if i % 10000 == 0:
            logging.info('Performing Anderson-Darling in element {0} out of {1}  [ {2:0.1f}% ] '.format(i+1, M, 100*float(i+1)/M))
        x = data[:,i]
        try: 
            anders = anderson(x)
            A = anders.statistic
            crit = anders.critical_values
            sig = anders.significance_level
            pval = 1.0
            for c, g in zip(crit, sig):
                if A >= c:
                    pval = g / 100.
        except:
            logging.debug('Anderson-Darling failed in element {}'.format(i+1))
            A = 0.0 
            pval = 0.0 
        assign_to_results(results,'AndersonDarlingStatistic',A,i,M,0.0)
        assign_to_results(results,'pval',pval,i,M,0.0)
        if pval > 0 and pval < 0.05:
            l_pval = -1*np.log10(pval[pval>0])
        else:
            l_pval = 0.0
        assign_to_results(results,'log10pval_trendP',l_pval,i,M,0.0)
    return results 
    
def elementwise_pearson(data, covariate):                
    '''Evaluate the Pearson correlation coefficient and associated hypothesis test on each element of a rectangular array.
    
    Parameters 
    ----------
    data : numpy.ndarray 
        A ndarray of shape (n, m) where n is the number of subjects and m is the number of elements. 
    covariate : numpy.ndarray 
        A ndarray of shape (n, ) where n is the number of subjects. 
    Returns
    -------
    results
        An OrderedDict of elementwise pearson correlation results. 
    '''
    logging.debug('diciphr.statistics.elementwise.elementwise_pearson')
    n, M = data.shape
    #Initialize ...
    results = OrderedDict()
    for i in range(M):
        logging.debug('Calculating Pearson r in element {} out of {}'.format(i+1, M))
        g = data[:,i]
        try: 
            r, pval = pearsonr(g, covariate)
        except ValueError:
            logging.debug('Pearson r failed in element {}'.format(i+1))
            r = 0.0 
            pval = 1.0 
        assign_to_results(results,'r',r,i,M,0.0)
        assign_to_results(results,'pval',pval,i,M,1.0)
        assign_to_results(results,'qval',1.0,i,M,1.0)
        assign_to_results(results,'n',n,i,M,0.0)
    # FDR Correction
    results['qval'] = fdr(results['pval'])
    trendP_result = results['r'].copy()
    trendP_result[results['pval'] > 0.05] = 0.0 
    results['r_trendP'] = trendP_result
    fdrQ_result = results['r'].copy()
    fdrQ_result[results['qval'] > 0.05] = 0.0  
    results['r_fdrQ'] = fdrQ_result
    return results 

def elementwise_spearman(data, covariate):                
    '''Evaluate the Spearman correlation coefficient and associated hypothesis test on each element of a rectangular array.
    
    Parameters 
    ----------
    data : numpy.ndarray 
        A ndarray of shape (n, m) where n is the number of subjects and m is the number of elements. 
    covariate : numpy.ndarray 
        A ndarray of shape (n, ) where n is the number of subjects. 
    Returns
    -------
    results
        An OrderedDict of elementwise Spearman correlation results. 
    '''
    logging.debug('diciphr.statistics.elementwise.elementwise_spearman')
    n, M = data.shape
    #Initialize ...
    results = OrderedDict()
    for i in range(M):
        if i % 10000 == 0:
            logging.info('Performing spearman correlation in element {0} out of {1}  [ {2:0.1f}% ] '.format(i+1, M, 100*float(i+1)/M))
        g = data[:,i]
        try: 
            r, pval = spearmanr(g, covariate)
        except ValueError:
            logging.debug('Spearman r failed in element {}'.format(i+1))
            r = 0.0 
            pval = 1.0 
        assign_to_results(results,'r',r,i,M,0.0)
        assign_to_results(results,'pval',pval,i,M,1.0)
        assign_to_results(results,'qval',1.0,i,M,1.0)
        assign_to_results(results,'n',n,i,M,0.0)
    # FDR Correction
    results['qval'] = fdr(results['pval'])
    trendP_result = results['r'].copy()
    trendP_result[results['pval'] > 0.05] = 0.0 
    results['r_trendP'] = trendP_result
    fdrQ_result = results['r'].copy()
    fdrQ_result[results['qval'] > 0.05] = 0.0  
    results['r_fdrQ'] = fdrQ_result
    return results 
    
def results_to_dataframe(results, header=None):     
    if header is None:
        header = ["{0:03d}".format(i) for i in range(len(results.values()[0]))]
    df = pd.DataFrame(index=results.keys(),columns=header)
    for k in results:
        df.loc[k,:] = results[k]
    return df.transpose()

def results_to_adjacency(results, diagonal=False):
    ret = OrderedDict()
    for k in results:
        if k in ['df_resid', 'df_model', 'n']:
            continue 
        if k.startswith('p_') or k.startswith('q_'):
            mat = ut_to_square(results[k], default_value=1, diagonal=diagonal)
        else:
            mat = ut_to_square(results[k], default_value=0, diagonal=diagonal)
        ret[k] = mat 
    return ret 

def results_to_niftis(results, mask_im):
    ret = OrderedDict()
    mask_data = mask_im.get_data() > 0
    for k in results:
        if k in ['df_resid', 'df_model', 'n']:
            continue 
        if k.startswith('p_') or k.startswith('q_'):
            dat = np.ones(mask_data.shape)
        else:
            dat = np.zeros(mask_data.shape)
        dat[mask_data] = results[k]
        nifti_im = nifti_image(dat, mask_im.affine)
        ret[k] = nifti_im
    return ret 