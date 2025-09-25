import logging
import numpy as np
import pandas as pd
import patsy
from diciphr.utils import DiciphrException, logical_or, logical_and, is_string, force_to_list

class DiciphrStatsException(DiciphrException): pass

def arrays_same_length(*args):
    a0=args[0]
    a0=np.array(a0)
    N=a0.shape[0]
    for a in args[1:]:
        a = np.array(a)
        if a.shape[0] != N:
            return False
    return True
        
def is_number(a):
    try:
        int(a)
    except ValueError:
        return False
    return True and not is_string(a)
    
def is_integer(a):
    if is_number(a):
        return a == int(a) and '.' not in str(a)

def is_list(a):
    return isinstance(a, (list, tuple, np.ndarray))
    
def is_numpy_array(a):
    return isinstance(a, np.ndarray)
        
def filter_cohort(cohort, expression, data=None):
    ''' Filter a cohort by an expression, e.g. Age>20 '''
    logging.debug("diciphr.statistics.utils.filter_cohort")
    selected = np.array([True for i in range(len(cohort.index))])
    if '=' in expression:
        col, vals = expression.split('=')
        vals = vals.split(',')
        selected = logical_or(*[ cohort[col] == v for v in vals ])
        logging.debug(selected)
    elif '!' in expression:
        col, vals = expression.split('!')
        vals = vals.split(',')
        selected = [ cohort[col] != v for v in vals ]
        selected = logical_and(*selected)
    elif '<' in expression:
        if len(expression.split('<')) == 3:
            lower, col, upper = expression.split('<')
            selected = logical_and(cohort[col] > lower, cohort[col] < upper)
        else:
            col, upper = expression.split('<')
            selected = cohort[col] < upper
    elif '>' in expression: 
        if len(expression.split('>')) == 3:
            upper, col, lower = expression.split('>')
            selected = logical_and(cohort[col] > lower, cohort[col] < upper)
        else:
            col, lower = expression.split('>')
            selected = cohort[col] > lower
    if data is not None:
        return cohort.loc[selected,:], data[selected, ...]
    else:
        return cohort.loc[selected,:]
        
# def make_design(cohort, formula, filename_template='', centralize=[], filter=[]):
def make_design(cohort, formula, centralize=[], filter=[], treatments={}, intercept=False):
    logging.debug("diciphr.statistics.utils.make_design")
    logging.info("Dataframe has {} entries.".format(len(cohort)))
    # User defined filters
    centralize = force_to_list(centralize)
    filter = force_to_list(filter)
    for expr in filter:
        logging.info("Applying filter {}".format(expr))
        cohort = filter_cohort(cohort, expr)
    if len(filter) > 0:
        logging.info("Filters complete. Dataframe now has {} entries.".format(len(cohort)) )
    if treatments:
        for key in treatments:
            logging.info('Implementing treatment {} with baseline level {}'.format(key, treatments[key]))
            formula = formula.replace(key, "C("+key+", Treatment('"+treatments[key]+"'))")
    # Only use the columns we need 
    possible_columns = []
    # for w in cohort.columns:
        # if w in formula.replace("*","+").replace(":","+").replace("(","+").replace(")","+").split("+"): #ugly 
            # possible_columns.append(w)
    for w in formula.replace("*","+").replace(":","+").replace("(","+").replace(")","+").split("+"):
        if w.strip() in cohort.columns:
            possible_columns.append(w)
    logging.debug('Ascertained columns from formula: {}'.format(possible_columns))
    cohort = cohort[possible_columns]
    # Drop any NaNs
    n = len(cohort)    
    cohort = cohort.dropna(axis=0)
    if len(cohort) < n:
        logging.info("Dropped {} rows containing NaN from the design".format(n - len(cohort)))
    # Centralize desired variables
    for var in centralize:
        logging.info("Centralize {}".format(var))
        cohort.loc[:, var] -= np.mean(cohort[var])
    # Create the design 
    logging.info('Patsy formula: {}'.format(formula))
    design = patsy.dmatrix(formula, data=cohort, return_type='dataframe')
    logging.debug('Design: \n{}'.format(design))
    #first one is intercept, delete it
    if not intercept:
        columns = design.columns[1:]
        design = design[columns]
    return design
 
def fdr(pvalues, correction_type='Benjamini-Hochberg'):
    """
    fdr(pvalues, correction_type='Benjamini-Hochberg'):
    
    Returns an adjusted pvalue array same shape as the input.
    
    http://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python
    """
    
    #input a series get a series of results
    if isinstance(pvalues,pd.Series):
        pvalues_dataframe_out = pd.Series(index=pvalues.index,dtype=object)
        for ind in pvalues.index:
            pvalues_dataframe_out[ind] = fdr(pvalues[ind],correction_type=correction_type)
        return pvalues_dataframe_out
    
    #retain a copy of the data for reshaping to original format
    pvalues_orig = np.array(pvalues)
    pvalues_out = np.ones((np.size(pvalues_orig),))  #1d array
    
    #grab only those p-values that are not exactly equal to 1
    # (pvalues are 1 when the model was not fit and the element was skipped over)
    w = np.where(pvalues_orig.flatten() < 1)[0]
    # asarray([_i if _p < 1 else None for _i,_p in enumerate(pvalues_orig.flatten())])
    pvalues = pvalues_orig.flatten()[w]
    
    n = pvalues.shape[0]
    new_pvalues = np.empty((n,))
    if correction_type=='Bonferroni':
        new_pvalues = n*pvalues
        new_pvalues[new_pvalues>1]=1
    elif correction_type=='Bonferroni-Holm':
        values = [(pvalue,i) for i,pvalue in enumerate(pvalues)]
        values.sort()
        for rank,vals in enumerate(values):
            new_pvalues[vals[1]] = (n-rank)*vals[0]
    elif correction_type=='Benjamini-Hochberg':
        values = [(pvalue,i) for i,pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i,vals in enumerate(values):
            rank = n-i
            pvalue,index = vals
            new_values.append((n/rank)*pvalue)
        for i in range(0,int(n)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i,vals in enumerate(values):
            new_pvalues[vals[1]] = new_values[i]
    
    #fill in values
    pvalues_out[w] = new_pvalues
    
    return pvalues_out.reshape(pvalues_orig.shape)

def ICC_rep_anova(Y):
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    
    Borrowed from nipy.nipype.algorithms.icc.py 
    '''
    from numpy import ones, kron, mean, eye, hstack, dot, tile
    from scipy.linalg import pinv
    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = mean(Y)
    SST = ((Y - mean_Y)**2).sum()

    # create the design matrix for the different levels
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))  # sessions
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = hstack([x, x0])

    # Sum Square Error
    predicted_Y = dot(dot(dot(X, pinv(dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals**2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((mean(Y, 0) - mean_Y)**2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) /
    #            (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    return ICC, r_var, e_var, session_effect_F, dfc, dfe
    
def standalone_colorbar(cmap='seismic', clims=[-1,1], orientation='horizontal', length=1, aspect=4, dpi=600):
    # Create a standalone colorbar inside a figure. 
    import matplotlib.pyplot as plt
    if orientation.lower() == 'horizontal':
        figsize=(length, length/aspect)
        adjust_kw={'bottom':0.5}
    else:
        figsize=(length/aspect, length)
        adjust_kw={'right':0.5}
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(**adjust_kw)
    my_cmap = mpl.cm.get_cmap(cmap)
    my_norm = mpl.colors.Normalize(vmin=clims[0], vmax=clims[1])
    sm = mpl.cm.ScalarMappable(norm=my_norm, cmap=my_cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation=orientation, label='')
    if 0 > clims[0] and 0 < clims[1]:
        cb.set_ticks([clims[0], 0, clims[1]])
        cb.ax.set_xticklabels([str(clims[0]),'0',str(clims[1])], fontsize=5, color='k')
    else:
        cb.set_ticks([clims[0], clims[1]])
        cb.ax.set_xticklabels([str(clims[0]),str(clims[1])], fontsize=5, color='k')
    return fig 
   
def values2colors(values, cmap, vmax=None):
    # Convert some values to color tuples using a cmap, useful for coloring a barplot. 
    if vmax is None:
        vmax = np.max(np.abs(values))
    vmin = -1*vmax
    v = np.clip(values, vmin, vmax)
    v = (v-vmin)/(vmax-vmin)
    return cmap(v)