#! /usr/bin/env python

import os, argparse, sys
import re 
import pandas as pd
import numpy as np 
import nibabel as nib
import patsy 
import logging
from ..utils import check_inputs, make_dir, protocol_logging
from ..statistics.harmonization import combat, covbat
from ..nifti_utils import ( nifti_image, read_nifti, 
            strip_ext, strip_nifti_ext, get_nifti_ext )

DESCRIPTION = '''
    Run ComBat or CovBat harmonization on NiFTI or csv data. 
'''
PROTOCOL_NAME='ComBat' 

def parseCommandLine(argv=None):
    # create arguments for the inputs
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-f', type=str, dest='filename_template', required=True,
                help='Path to a csv of measurements with subject ID in the leftmost column, OR '+
                    'path to template space nifti files, with {s} in place of subject ID.')
    p.add_argument('-o',  type=str, dest='outdir', required=True, 
                help='The output directory.')
    p.add_argument('-c', '-cohort', type=str, dest='cohort', required=True,
                help='CSV file of the cohort, with subject IDs matching registered nifti filenames or csv-format data in first column.')
    p.add_argument('-I', '-id', type=str, default=None, required=False, 
                help='If using a different column in the cohort csv for the subject IDs, provide it')
    p.add_argument('-m', '-mask', type=str, default=None, dest='mask', 
                help='Mask in template space to use for analysis. If not provided, will find voxels with nonzero variance across sample.')    
    p.add_argument('-v', '--covbat', action='store_true', dest='covbat', 
                help='Run CovBat instead of default ComBat.')
    p.add_argument('-V', '-covariates', type=str, default=None, dest='formula', 
                help='Patsy-like formula for the covariates (e.g "Age+Sex").')
    p.add_argument('-C', '-continuous', type=str, default=None, nargs="*", dest='continuous', 
                help='Column name(s) of the continuous variables as opposed to categorical (e.g. "Age").')                
    p.add_argument('-S', '-site', type=str, default='Site', dest='site', 
                help='Column name encoding the site or study variable. Default is "Site".')
    p.add_argument('-F',  type=str, default=None, dest='out_template', 
                help='If harmonizing nifti images, filename template with {s} for the output files (basename only). Will default to input basename.')
    # not implemented in covbat 
    # parser.add_argument('-r', '-refbatch', type=str, default=None, dest='refbatch', 
                # help='Site to be used as the batch reference, if any.')
    p.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    p.add_argument('--logfile', action='store', metavar='log', dest='logfile', 
                    type=str, required=False, default=None, 
                    help='A log file. If not provided will print to stderr.'
                    )
    args = p.parse_args(argv)
    try:
        ext = get_nifti_ext(os.path.basename(args.filename_template))
        args.nifti = True 
    except:
        # not nifti - csv mode 
        args.nifti = False 
    if args.nifti and args.out_template is None:
        basename = strip_nifti_ext(os.path.basename(args.filename_template))
        ext = get_nifti_ext(os.path.basename(args.filename_template))
        args.out_template = basename+'_COMBAT.'+ext
    elif args.out_template is None:
        basename = strip_ext(os.path.basename(args.filename_template))
        args.out_template = basename+'_COMBAT.csv'
    return args
 
def load_data_niftis(cohort, filename_template, mask_file=None):
    # load the data from subjects in cohort 
    # Raise error if any are missing 
    # Apply mask from nifti or from variance 
    # return nifti image of the mask for future 
    filenames = [ filename_template.format(s=s) for s in cohort.index ] 
    exists = np.array([ os.path.exists(f) for f in filenames])
    cohort = cohort.loc[exists,:]
    data = [ read_nifti(filename_template.format(s=s)).get_fdata() for s in cohort.index ]
    data = np.asarray(data)
    if mask_file:
        mask_img = read_nifti(mask_file)
    else:
        affine = read_nifti(filenames[0]).affine
        mask_img = nifti_image((np.var(data, axis=0) > 0)*1, affine)
    # convert data to 2d array of shape N, v  (subjects, voxels)
    data = data[:, mask_img.get_fdata() > 0]
    data = pd.DataFrame(data, index=cohort.index)
    return data.transpose(), mask_img, cohort
   
def load_data_csv(cohort, filename, index_column=None, rois=[]):
    # load the data from subjects in cohort 
    # Raise error if any are missing 
    # Apply mask from nifti or from variance 
    # return nifti image of the mask for future 
    data = pd.read_csv(filename)
    if index_column is None:
        index_column = data.columns[0]
    data = data.set_index(index_column)
    subjects = cohort.index.intersection(data.index)
    cohort = cohort.loc[subjects,:]
    data = data.loc[subjects,:]
    if rois:
        data = data.loc[:,rois]
    rois = list(data.columns)
    # data = data.get_values().astype(float)
    return data.transpose(), rois, cohort
    
def split_formula(formula):
    # return a list of the words in the formula 
    words = list(filter(lambda c: bool(c), re.split('\)|\(|\+|\*|\:|\ ', formula)))
    return words 
    
def prepare_cohort(cohortcsv, site, formula, na_values = ['.','NAN','NaN','nan','Nan','NA','na','n/a','N/A',' ']):
    # Read cohort, grab only columns needed, remove any missing data
    cohort = pd.read_csv(cohortcsv, na_values=na_values)
    cohort = cohort.set_index(cohort.columns[0])
    covariates = split_formula(formula)
    if site in covariates:
        raise ValueError('Site column ({0}) cannot be in covariate formula'.format(site))
    cohort = cohort[[site]+covariates]
    cohort = cohort.dropna()
    return cohort 
    
def covbat_post_niftis(harmonized_data, cohort, mask_img, filename_template):
    mask = mask_img.get_fdata() > 0
    for i, subject in enumerate(cohort.index):
        sub_data = np.zeros(mask_img.shape, dtype=np.float32)
        sub_data[mask] = np.asarray(harmonized_data)[:,i]
        sub_im = nifti_image(sub_data, mask_img.affine)
        sub_im.to_filename(filename_template.format(s=subject))
        logging.info(filename_template.format(s=subject))
 
def main(argv):
    args = parseCommandLine(argv)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug) 
    logging.info("Read cohort file")
    cohort = prepare_cohort(args.cohort, args.site, args.formula)
    os.makedirs(args.outdir, exist_ok=True)
    logging.info("Load data")
    if args.nifti:
        data, mask_img, cohort = load_data_niftis(cohort, args.filename_template, args.mask)
        mask_img.to_filename(os.path.join(args.outdir, 'covbat_mask.nii.gz'))
    else:
        data, rois, cohort = load_data_csv(cohort, args.filename_template)
    logging.info("Data loaded successfully")
    cohort.to_csv(os.path.join(args.outdir, 'covbat_cohort.csv'), index_label='Subject')
    # check for sites 
    logging.info(args.site)
    for _site in cohort[args.site].unique():
        _count = (cohort[args.site] == _site).sum()
        if _count < 2:
            raise ValueError("Cannot harmonize - data from site {} has {} observations".format(_site, _count))
    logging.info("Prepare covariate dataframe") 
    mod = patsy.dmatrix(args.formula, cohort, return_type="dataframe")
    mod.to_csv(os.path.join(args.outdir, 'mod.csv'), index_label='Subject')
    #continuous columns are those which contain string matching user input (includes interactions)
    continuous = []
    for c in args.continuous:
        for col in mod.columns:
            if c in col.split(':') and col not in continuous:
                continuous.append(col)
    logging.info("Prepare site dataframe") 
    batch = cohort[[args.site]]
    batch['index'] = cohort[args.site].apply(lambda x: list(cohort[args.site].unique()).index(x)+1)
    batch.to_csv(os.path.join(args.outdir, 'batch.csv'), index_label='Subject')
    logging.info("Run neuroCombat")
    if args.covbat:
        harmonized_data, gamma_hat, gamma_star, delta_hat, delta_star = covbat(data, batch['index'], model=mod, numerical_covariates=continuous, return_estimates=True)
    else:
        harmonized_data, gamma_hat, gamma_star, delta_hat, delta_star = combat(data, batch['index'], model=mod, numerical_covariates=continuous, return_estimates=True)
    if args.nifti:
        logging.info("Write results to Niftis ")
        covbat_post_niftis(harmonized_data, cohort, mask_img, os.path.join(args.outdir, args.out_template))
    else:
        # covbat_df = pd.DataFrame(harmonized_data.T, index=cohort.index, columns=rois)
        harmonized_data.transpose().to_csv(os.path.join(args.outdir, args.out_template), index=True, index_label='Subject')
        np.savetxt("{0}/gamma_hat.csv".format(args.outdir),gamma_hat,delimiter=',')
        np.savetxt("{0}/gamma_star.csv".format(args.outdir),gamma_star,delimiter=',')
        np.savetxt("{0}/delta_hat.csv".format(args.outdir),delta_hat,delimiter=',')
        np.savetxt("{0}/delta_star.csv".format(args.outdir),delta_star,delimiter=',')
    logging.info("Done")

if __name__ == '__main__':
    main(sys.argv[1:])
