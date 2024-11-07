# -*- coding: utf-8 -*-

import os, sys, shutil, logging
import numpy as np
from collections import OrderedDict
from scipy.stats import mannwhitneyu, ttest_ind, norm, gamma
from ..resources import labels_86, labels_87, labels_wm
from ..statistics.utils import fdr, dmatrix
from .connmat_utils import symmetric_mat_from_upper_mask

##########################
### Plotting Functions ###
##########################
def sample_edge_distribution(mat_array, i_index, j_index, show=False, roi_names=[], pdf='gamma', log_scale=False):
    ''' 
    mat_array shape is n_subjects,n_nodes,n_nodes
    '''
    import matplotlib.pyplot as plt
    edges=mat_array.copy()[:,i_index,j_index]
    if log_scale:
        edges = np.log(edges + 1)
    plt.hist(edges, bins=12, normed=True, alpha=0.25, color='g')
    xmin, xmax = plt.xlim()
    x=np.linspace(xmin, xmax, 1000)
    if pdf == 'normal':
        mu, std = norm.fit(edges)
        p = norm.pdf(x, mu, std)
    elif pdf == 'gamma':
        alpha, loc, beta = gamma.fit(edges)
        p = gamma.pdf(x, alpha, loc, beta)
    ax = plt.plot(x,p,'r-')
    if roi_names:
        plt.title('({0} - {1})'.format(roi_names[i_index],roi_names[j_index]), fontsize=8)
    else:
        plt.title('({0} - {1})'.format(i_index, j_index), fontsize=8)
    _med = np.median(edges)
    _max = np.max(edges)
    if _med > 0:
        plt.xticks([xmin,_med,_max],["{0:.1f}".format(xmin),"{0:.1f}".format(_med),"{0:.1f}".format(_max)])
    else:
        plt.xticks([xmin,_max],["{0:.1f}".format(xmin),"{0:.1f}".format(_max)])
    plt.yticks([])
    plt.xlim(xmin, xmax)
    if show:
        plt.show()
        
def connmat_edge_distribution(mat, thresh_low=0, thresh_high=None, log_scale=False, show=False):
    import matplotlib.pyplot as plt
    edges = mat.copy()[np.triu_indices(mat.shape[0],1)]
    num_edges = edges.size
    edges = edges[edges>thresh_low]
    if thresh_high:
        edges = edges[edges<thresh_high]
    else:
        thresh_high = np.max(edges)
    #edges=edges/float(num_edges)
    if log_scale:
        n, bins, patches = plt.hist(np.log10(edges+1), 50, facecolor='green', alpha=0.25)
        plt.xticks(np.log10([11,101,1001,10001]),[10,100,1000,10000])
        plt.xlim(0,np.log10(thresh_high+1))
    else:
        n, bins, patches = plt.hist(edges, 50, facecolor='green', alpha=0.25) # normed=1,
    plt.xlabel('Edge weight')
    plt.ylabel('Count')
    if show:
        plt.show()
    return n, bins, patches   

###########################
### Edgewise Statistics ###
###########################
class EdgewiseResultsContainer(object):
    def __init__(*kwargs):
        self.data = OrderedDict()
        for key, value in kwargs.items():
            if key == 'upper_triangular_mask':
                self.upper_triangular_mask = upper_triangular_mask.copy()
            else:
                self.data[key] = value
            
    def set_edge_value(key, edges, upper_triangular_mask=None, fill_value=0.0):
        if upper_triangular_mask:
            edges = symmetric_mat_from_upper_mask(edges, upper_triangular_mask, fill_value=fill_value)
        self.data[key] = edges
        
    def threshold_edge_value(key, edges, threshold_value, threshold_edges=None, upper_triangular_mask=None, fill_value=0.0, absvalue=True):
        if upper_triangular_mask:
            edges = symmetric_mat_from_upper_mask(edges, upper_triangular_mask, fill_value=fill_value)
        if threshold_edges is None:
            threshold_edges = edges.copy()
        if absvalue:
            mask = np.abs(threshold_edges) < threshold_value
        else:
            mask = threshold_edges < threshold_value
        edges[mask] = fill_value
        self.set_edge_value(key, edges)
        
    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __delitem__(self, key):
        del self.data[key]

    def clear(self):
        return self.data.clear()

    def copy(self):
        return self.data.copy()

    def has_key(self, k):
        return k in self.data

    def update(self, *args, **kwargs):
        return self.data.update(*args, **kwargs)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def pop(self, *args):
        return self.data.pop(*args)

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        return iter(self.data)
        
def edgewise_ols(edges_sample_array, upper_triangular_mask, formula="Group", dataframe=[], comparison_formula=None, logscale=False):
    import statsmodels.formula.api as smf
    '''
    Edgewise OLS on connmats.

    edges_sample_array: numpy ndarray of N x M edges
    upper_triangular_mask: numpy ndarray of network mask, shape (numnodes,numnodes), sums to M
    formula: a patsy formula for the ols
    dataframe: a pandas dataframe, or a list of 0's and 1's or N labels. 

    returns EdgewiseResultsContainer
        keys organized by statistic and covariate name, if applicable.
    '''
    nsubjs, __ = edges_sample_array.shape
    numnodes, __ = upper_triangular_mask.shape
    zeros=np.zeros(upper_triangular_mask.sum())
    ret = EdgewiseResultsContainer()
    ret['F-value'] = zeros.copy()
    ret['F-pvalue'] = zeros + 1.0
    ret['F-qvalue'] = zeros + 1.0
    if comparison_formula:
        ret['F-value_CompareModels'] = zeros.copy()
        ret['F-pvalue_CompareModels'] = zeros + 1.0
        ret['F-qvalue_CompareModels'] = zeros + 1.0
    ret['Adjusted-R2'] = zeros.copy()
    if logscale: 
        edges_sample_array = np.log(edges_sample_array+1)
    if not isinstance(dataframe, pd.DataFrame):
        dataframe = pd.DataFrame(headers=["Group"], data=dataframe)
    dataframe = dataframe.copy()
    X = dmatrix(formula, dataframe, return_type='dataframe')
    for i, y in enumerate(edges_sample_array.transpose()):
        model = smf.OLS(y, X)
        results = model.fit()
        ret['F-value'][i] = results.fvalue()
        ret['F-pvalue'][i] = results.f_pvalue()
        ret['Adjusted-R2'][i] = results.rsquared_adj()
        df = results.df_resid
        if comparison_formula: 
            model_compar = smf.ols(formula=comparison_formula, data=dataframe)
            results_compar = model_compar.fit()
            ret['F-value_CompareModels'][i], ret['F-pvalue_CompareModels'][i], __ = results.compare_f_test(results_compar)
            
        for name in results.exog_names:
            if not ret.has_key('tval_'+name):
                ret['tval_'+name] = zeros.copy()
                ret['coeff_'+name] = zeros.copy()
                ret['dval_'+name] = zeros.copy()
                ret['pval_'+name] = zeros + 1.0
                ret['qval_'+name] = zeros + 1.0
            ret['tval_'+name][i] = results.tvalues[name]
            ret['pval_'+name][i] = results.pvalues[name]
            ret['coeff_'+name][i] = np.mean(results.conf_int(cols=results.pvalues[name]))
            ret['dval_'+name][i] = 2*ret['tval_'+name][i] / np.sqrt(df)
    #fdr 
    for key in ret.keys():
        if 'pval' in key:
            ret[key.replace('pval','qval')][:] = fdr(ret[key])
        ret.set_edge_value(key, ret[key], upper_triangular_mask)
    return ret
    
def edgewise_mannwhitneyu(edges_sample_array, upper_triangular_mask, groups):
    '''
    Edgewise mann whitney U on connmats.
    
    edges_sample_array: numpy ndarray of N x M edges 
    upper_triangular_mask: numpy ndarray of network mask, shape (numnodes,numnodes), sums to M 
    groups: numpy ndarray shape (N,) of zeros (controls) and ones (patients) 
    
    returns
    uvals: numnodes,numnodes array of U test statistics
    dvals: numnodes,numnodes array of effect size, with sign
    pvals: numnodes,numnodes array of p values
    qvals: numnodes,numnodes array of FDR-corrected q values
    
    '''
    nsubjs, __ = edges_sample_array.shape
    numnodes, __ = upper_triangular_mask.shape
    uvals = np.zeros(upper_triangular_mask.sum())
    dvals = np.zeros(upper_triangular_mask.sum())
    pvals = np.zeros(upper_triangular_mask.sum())
    qvals = np.zeros(upper_triangular_mask.sum())
    edges_sample_array = np.log(edges_sample_array+1)
    for i,edge_sample in enumerate(edges_sample_array.transpose()):
        logging.debug(' Mann-Whitney U in edge {} / {}'.format(i+1, upper_triangular_mask.sum()))
        data0 = edge_sample[groups == 0]
        data1 = edge_sample[groups == 1]
        u_1, p_1 = mannwhitneyu(data0,data1,alternative='two-sided')
        sign = np.sign(np.median(data1) - np.median(data0)) 
        # rank-biserial correlation
        e_1 = 1 - ((2*u_1)/float(data0.size * data1.size))
        uvals[i] = u_1
        dvals[i] = e_1
        pvals[i] = p_1
    qvals = fdr(pvals)
    uvals = symmetric_mat_from_upper_mask(uvals, upper_triangular_mask, fill_value=0.0)
    dvals = symmetric_mat_from_upper_mask(dvals, upper_triangular_mask, fill_value=0.0)
    pvals = symmetric_mat_from_upper_mask(pvals, upper_triangular_mask, fill_value=1.0)
    qvals = symmetric_mat_from_upper_mask(qvals, upper_triangular_mask, fill_value=1.0)
    return uvals, dvals, pvals, qvals    
    
def edgewise_ttest(edges_sample_array, upper_triangular_mask, groups):
    '''
    Edgewise ttest on connmats.
    
    edges_sample_array: numpy ndarray of N x M edges 
    upper_triangular_mask: numpy ndarray of network mask, shape (numnodes,numnodes), sums to M 
    groups: numpy ndarray shape (N,) of zeros (controls) and ones (patients) 
    
    returns
    tvals: numnodes,numnodes array of t test statistics
    dvals: numnodes,numnodes array of effect size, with sign
    pvals: numnodes,numnodes array of p values
    qvals: numnodes,numnodes array of FDR-corrected q values
    
    '''
    nsubjs, __ = edges_sample_array.shape
    numnodes, __ = upper_triangular_mask.shape
    tvals = np.zeros(upper_triangular_mask.sum())
    dvals = np.zeros(upper_triangular_mask.sum())
    pvals = np.zeros(upper_triangular_mask.sum())
    qvals = np.zeros(upper_triangular_mask.sum())
    edges_sample_array = np.log(edges_sample_array+1)
    for i,edge_sample in enumerate(edges_sample_array.transpose()):
        logging.debug('t-test in edge {} / {}'.format(i+1, upper_triangular_mask.sum()))
        data0 = edge_sample[groups == 0]
        data1 = edge_sample[groups == 1]
        n = len(groups)
        t_1, p_1 = ttest_ind(data0,data1,equal_var=True)
        sign = np.sign(np.median(data1) - np.median(data0)) 
        # rank-biserial correlation
        d_1 = 2*t_1/np.sqrt(n-1)
        tvals[i] = t_1
        dvals[i] = d_1
        pvals[i] = p_1
    qvals = fdr(pvals)
    tvals = symmetric_mat_from_upper_mask(tvals, upper_triangular_mask, fill_value=0.0)
    dvals = symmetric_mat_from_upper_mask(dvals, upper_triangular_mask, fill_value=0.0)
    pvals = symmetric_mat_from_upper_mask(pvals, upper_triangular_mask, fill_value=1.0)
    qvals = symmetric_mat_from_upper_mask(qvals, upper_triangular_mask, fill_value=1.0)
    return tvals, dvals, pvals, qvals   

def edgewise_ttest_max(edges_sample_array, groups):
    '''
    Edgewise ttest on connmats.
    
    edges_sample_array: numpy ndarray of N x M edges 
    upper_triangular_mask: numpy ndarray of network mask, shape (numnodes,numnodes), sums to M 
    groups: numpy ndarray shape (N,) of zeros (controls) and ones (patients) 
    
    returns
    tmax: maximum absolute value t statistic of the array 
    '''
    nsubjs, __ = edges_sample_array.shape
    edges_sample_array = np.log(edges_sample_array+1)
    ts=[]
    for i,edge_sample in enumerate(edges_sample_array.transpose()):
        data0 = edge_sample[groups == 0]
        data1 = edge_sample[groups == 1]
        n = len(groups)
        t_1, p_1 = ttest_ind(data0,data1,equal_var=True)
        ts.append(t_1)
    return np.max(np.abs(ts))

def permutation_test(edges_sample_array, upper_triangular_mask, groups, nperms=100):
    tvals, dvals, pvals, qvals = edgewise_ttest(edges_sample_array, upper_triangular_mask, groups)
    tdist=[]
    for i in range(nperms):
        if (i+1)%100 == 0:
            sys.stdout.write("Permutation {0} out of {1}\n".format(i+1,nperms))
            sys.stdout.flush()
        permgroups = groups[np.random.permutation(len(groups))]
        tdist.append(edgewise_ttest_max(edges_sample_array, permgroups))
    tdist = np.array(sorted(tdist))
    tcrit = np.percentile(tdist, 95)
    tvals[np.abs(tvals) < tcrit] = 0
    dvals[np.abs(tvals) < tcrit] = 0
    pvals[np.abs(tvals) < tcrit] = 1
    qvals = np.ones(pvals.shape)
    qvals[tvals!=0] = np.array([np.sum(tdist>np.abs(t)) for t in tvals[tvals!=0]])/float(nperms)
    return tvals, dvals, pvals, qvals
 