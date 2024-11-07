#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
import re 
import numpy as np
import pandas as pd 
import pickle 
from diciphr.utils import check_inputs, make_dir, protocol_logging
from diciphr.statistics.elementwise import elementwise_ols, results_to_dataframe
from diciphr.statistics.utils import filter_cohort
from diciphr.nifti_utils import replace_labels, read_nifti, write_nifti
from diciphr.oscar import Oscar 
    
DESCRIPTION = '''
    ROI Stats OLS analysis
'''

PROTOCOL_NAME='OLS_Analysis'
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','--data', action='store',metavar='data.csv',dest='datafile',
                    type=str, required=True, 
                    help='The ROIstats csv file'
                    )
    p.add_argument('-c','--cohort', action='store',metavar='cohort.csv',dest='cohortfile',
                    type=str, required=True, 
                    help='The cohort csv file'
                    )
    p.add_argument('-o','--outdir', action='store',metavar='str',dest='outdir',
                    type=str, required=True, 
                    help='The output directory.'
                    )
    p.add_argument('-p','--prefix', action='store',metavar='str',dest='prefix',
                    type=str, required=False, default='',  
                    help='Optional prefix for the results CSV file to be saved within output directory. Default is "ols_results".'
                    )
    p.add_argument('-f','--formula', action='store',metavar='str',dest='formula',
                    type=str, required=True, 
                    help='The OLS analysis formula, e.g. Group+Age+Sex'
                    )
    p.add_argument('-r','--reduced_formula', action='store',metavar='str',dest='reduced_formula',
                    type=str, required=False, default='', 
                    help='A reduced OLS analysis formula, e.g. Age+Sex'
                    )
    p.add_argument('-C','--centralize', action='store',metavar='str',dest='centralize',
                    type=str, required=False, default=[], nargs="*", 
                    help='Covariate(s) to centralize before analysis. e.g. Age'
                    )
    p.add_argument('-F','--filter', action='store',metavar='str',dest='filters',
                    type=str, required=False, default=[], nargs="*", 
                    help='Filter(s) to apply before analysis, e.g. Site=Site1 Sex=Male Age>18, etc.'
                    )
    p.add_argument('-T','--treatment', action='store',metavar='str',dest='treatments',
                    type=str, required=False, default=[], nargs="*", 
                    help='Define which level in a categorical covariate is treated as baseline, e.g. Group=Control.'
                    )
    p.add_argument('-L','--features', action='store', metavar='str', dest='featuresfile', 
                    type=str, required=False, default=None, 
                    help='A text file containing the feature (e.g. ROI) column names in which to fit the models.'
                    )
    p.add_argument('-a','--atlas', action='store',metavar='str',dest='atlasfile',
                    type=str, required=False, default=None, 
                    help='A nifti file of roi labels. Results will be saved as nifti files with labels replaced by statistical results.'
                    )
    p.add_argument('-u','--underlay', action='store',metavar='str',dest='underlayfile',
                    type=str, required=False, default=None, 
                    help='A nifti file for underlay in screenshots of results.'
                    )
    p.add_argument('--slicetype', action='store', metavar='str',dest='slicetype',
                    type=str, required=False, default='c', 
                    help='Slice type for screenshots. Options: a/c/s/axial/coronal/sagittal'
                    )                
    p.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    p.add_argument('--logfile', action='store', metavar='log', dest='logfile', 
                    type=str, required=False, default=None, 
                    help='A log file. If not provided will print to stderr.'
                    )
    return p

    
class OLSAnalysis():
    def __init__(self, data, cohort, formula, reduced_formula='', features=[], 
                    atlas=None, filters=[], treatments={}, centralize=[]):
        self.data = data.copy()
        self.cohort = cohort.copy()
        self.formula1 = formula 
        self.formula2 = reduced_formula
        self.filters = filters
        self.treatments = treatments
        self.centralize = centralize
        self.results_df = None 
        self.atlas = None 
        self.exogs = [] 
        self.subjects = [] 
        if features:
            self.features = features
        else:
            self.features = list(data.columns)
        if atlas is not None:
            self.atlas = atlas
            self.labels = [int(a.split('_')[0][1:]) for a in self.features]
        # intersect data 
        self._intersect_data_cohort() 
    
    def _intersect_data_cohort(self):
        _formula_names = list(filter(lambda c: bool(c), re.split('\)|\(|\+|\*|\:|\ ', self.formula1)))
        _filter_names = [re.split('\=|\>|\<', f.replace(' ',''))[0] for f in self.filters]
        self.exogs = list(set(_formula_names + _filter_names))
        c = self.cohort[self.exogs]
        d = self.data[self.features]
        cin = c.index.name
        din = d.index.name
        c = c.reset_index().dropna().drop_duplicates().set_index(cin)
        d = d.reset_index().dropna().drop_duplicates().set_index(din)
        c = c.loc[c.index.intersection(d.index),:]
        d = d.loc[c.index,:].values
        
        # apply filters to cohort and data 
        for f in self.filters:
            logging.info('Applying filter to data ' + f)
            c, d = filter_cohort(c, f, data=d)
        
        # get all subjects who have data in cohort and rois
        self.subjects = list(c.index)
        self.cohort = c
        self.data = d
    
    def fit(self):
        self.results, self.models, self.residuals = elementwise_ols(self.data, self.cohort, self.formula1, self.formula2, 
                filters=self.filters, centralize=self.centralize, treatments=self.treatments)
        self.results_df = results_to_dataframe(self.results, self.features)
        self.residuals.columns = self.features
        self._dnames = list(filter(lambda c: c.startswith('d_'), self.results_df.columns))
        self._dnames.remove('d_Intercept')
        self._tnames = [ 't_'+k[2:] for k in self._dnames ]
        self._pnames = [ 'p_'+k[2:] for k in self._dnames ]
        self._qnames = [ 'q_'+k[2:] for k in self._dnames ]
        self._cnames = [ 'coeff_'+k[2:] for k in self._dnames ]
        self._bnames = [ 'b_'+k[2:] for k in self._dnames ]
        self._fnames = list(filter(lambda c: c.startswith('f_'), self.results_df.columns))
        self._fpnames = [ 'p_'+k[2:] for k in self._fnames ]
        self._fqnames = [ 'q_'+k[2:] for k in self._fnames ]
        return self.results_df 
        
    def get_cohensd_names(self):
        return self._dnames
        
    def get_tstat_names(self):
        return self._tnames
        
    def get_coeff_names(self):
        return self._cnames 
    
    def get_beta_names(self):
        return self._bnames 
        
    def get_pval_names(self):
        return self._pnames 
        
    def get_qval_names(self):
        return self._qnames 
        
    def get_model_names(self):
        return self._fnames
        
    def save_dataframe(self, filename):
        if self.results_df is not None:
            self.results_df.to_csv(filename)
            logging.info('Saved results to file ' + filename)
    
    def save_cohort(self, filename):
        self.cohort.to_csv(filename)
        logging.info('Saved cohort to file ' + filename)
        
    def save_models(self, filename):
        logging.info('Saved OLS models to file with Pickle ' + filename)
        with open(filename, 'wb') as fid:
            pickle.dump(self.models, fid)
            
    # convenient function 
    def get_nifti_result(self, key, sig='none', alpha=0.05):
        values = self.results_df[key]
        if sig == 'trendP':
            keyP = 'p_'+key[2:]
            pvals = (self.results_df[keyP]<=alpha)
        elif sig == 'fdrQ':
            keyP = 'q_'+key[2:]
            pvals = (self.results_df[keyP]<=alpha)
        else:
            pvals = np.ones(values.shape)
        values *= pvals
        return replace_labels(self.atlas, self.labels, values)
    
    def write_nifti_outputs(self, outdir):
        filenames = [] 
        if self.atlas is not None:
            for c in self.get_coeff_names():
                cr = c.replace(':','-x-')
                write_nifti(os.path.join(outdir, cr+'.nii.gz'), self.get_nifti_result(c))
                filenames.append(os.path.join(outdir, cr+'.nii.gz'))
            for c in self.get_cohensd_names() + self.get_beta_names() + self.get_tstat_names() + self.get_model_names():
                cr = c.replace(':','-x-')
                write_nifti(os.path.join(outdir, cr+'.nii.gz'), self.get_nifti_result(c))    
                write_nifti(os.path.join(outdir, cr+'_trendP.nii.gz'), self.get_nifti_result(c, sig='trendP'))    
                write_nifti(os.path.join(outdir, cr+'_fdrQ.nii.gz'),self.get_nifti_result(c, sig='fdrQ'))    
                filenames.append(os.path.join(outdir, cr+'.nii.gz'))
                filenames.append(os.path.join(outdir, cr+'_trendP.nii.gz'))
                filenames.append(os.path.join(outdir, cr+'_fdrQ.nii.gz'))
            write_nifti(os.path.join(outdir, 'R2.nii.gz'), self.get_nifti_result('R2'))
            filenames.append(os.path.join(outdir, 'R2.nii.gz'))
            write_nifti(os.path.join(outdir, 'R2_adj.nii.gz'),self.get_nifti_result('R2_adj'))
            filenames.append(os.path.join(outdir, 'R2_adj.nii.gz'))
        return filenames 
        
    def write_residual_outputs(self, csvfile):
        self.residuals.to_csv(csvfile)
        return csvfile
        
def oscarOutputs(filenames, atlas_img, underlay_img, slicetype, outdir):
    for fn in filenames:
        filebase = os.path.basename(fn[:-7])
        imgdata = np.abs(read_nifti(fn).get_fdata())
        try:
            m = np.percentile(imgdata[imgdata>0],99)
        except IndexError:
            m = 1 
        if filebase.startswith('t_'):
            clims = ['-9,9']
            cmap = 'jet'
        elif filebase.startswith('d_') or filebase.startswith('b_'):
            clims = ['-1,1']
            cmap = 'jet'
        elif filebase.startswith('f_'):
            clims = ['0,{0:0.1f}'.format(m)]
            cmap = 'viridis'
        elif filebase.startswith('R2'):
            clims = ['0,1']
            cmap = 'viridis'
        else:
            clims = ['-{0:0.1f},{0:0.1f}'.format(m)]
            cmap = 'jet'
        O = Oscar(underlay_img, [read_nifti(fn)], clims=clims, cmap=cmap, bgcolor='k')
        # get best slices 
        sliceindexdict={'a':2, 'c':1, 's':0}
        sliceindex = sliceindexdict[slicetype]
        w = np.where(underlay_img.get_fdata()>0)[sliceindex]
        if slicetype == 's':
            center = int(atlas_img.shape[sliceindex]/2)
        else:
            center = int((w.max()+w.min())/2)    
        spacing = int( atlas_img.header.get_zooms()[sliceindex]*(w.max()-w.min())/20 )
        O.slice_grid(slicetype, output_filebase=os.path.join(outdir, 'oscar_'+slicetype+'_'+filebase), 
                        nrows=3, ncols=3, center=center, spacing=spacing, title=filebase)
    O.slice_grid_view(slicetype, output_filebase=os.path.join(outdir, 'oscar'), 
                    nrows=3, ncols=3, center=center, spacing=spacing)
                        
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    make_dir(args.outdir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug) 
    check_inputs(args.cohortfile)
    check_inputs(args.datafile)
    if args.atlasfile:
        check_inputs(args.atlasfile, nifti=True)
        atlas_img = read_nifti(args.atlasfile)
        if args.underlayfile:
            check_inputs(args.underlayfile, nifti=True)
            underlay_img = read_nifti(args.underlayfile)
        else:
            underlay_img = atlas_img 
        if args.slicetype[0].lower() in ['a','c','s']:
            slicetype = args.slicetype[0].lower()
        else:
            raise ValueError('Invalid option to --slicetype')
    else:
        atlas_img = None
    treatments = dict([s.split('=') for s in args.treatments])

    logging.info('Read data from csv files')
    data = pd.read_csv(args.datafile, index_col=0, na_values=['.',' ','','NA','NaN','nan'])
    cohort = pd.read_csv(args.cohortfile, index_col=0, na_values=['.',' ','','NA','NaN','nan'])
    if args.featuresfile:
        features = [ a.strip() for a in open(args.featuresfile, 'r').readlines() ]
    else:
        features = [] 
        
    logging.info('Perform OLS analysis')
    A = OLSAnalysis(data, cohort, args.formula, reduced_formula=args.reduced_formula, features=features,
                    centralize=args.centralize, filters=args.filters, treatments=treatments, 
                    atlas=atlas_img)
    A.fit()
    logging.info('Save results to csv')
    if args.prefix:
        A.save_dataframe(os.path.join(args.outdir, args.prefix+'.csv'))
    else:
        A.save_dataframe(os.path.join(args.outdir, 'ols_results.csv'))
    A.save_cohort(os.path.join(args.outdir, 'cohort.csv'))
    A.save_models(os.path.join(args.outdir, 'models.pkl'))
    A.write_residual_outputs(os.path.join(args.outdir, 'residuals.csv')) 
    
    if atlas_img is not None:
        logging.info('Save Nifti results')
        filenames = A.write_nifti_outputs(args.outdir)
        
        logging.info('Oscar Nifti screenshots')
        oscarOutputs(filenames, atlas_img, underlay_img, slicetype, args.outdir)
                            
if __name__ == '__main__': 
    main(sys.argv[1:])
    
    
    