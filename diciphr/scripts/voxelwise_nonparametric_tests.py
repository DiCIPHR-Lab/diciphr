#! /usr/bin/env python

import os, sys, logging
from diciphr.nifti_utils import nifti_image
import nibabel as nib
import pandas as pd
from diciphr.statistics.stats_utils import filter_cohort
from diciphr.statistics.elementwise import ( results_to_niftis,
            elementwise_anderson_darling, elementwise_mannwhitneyu )
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from numpy import asarray 

DESCRIPTION = '''
    Perform nonparametric tests per voxel.
'''

PROTOCOL_NAME='Voxelwise_Nonparametric_Tests'
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-f', action='store', metavar='filename', dest='filename_template',
                    type=str, required=True,
                    help='The data filename template, with {s} replaced by subject ID.'
    )
    p.add_argument('-m', action='store', metavar='mask', dest='mask',
                    type=str, required=True,
                    help='A nifti mask where to perform the statistics.'
    )
    p.add_argument('-o', action='store', metavar='outbase', dest='outbase',
                    type=str, required=True,
                    help='The output basename. Results will be written to {outbase}_U.nii.gz, etc.'
    )
    p.add_argument('-c', action='store', metavar='csv', dest='cohort',
                    type=str, required=True,
                    help='The cohort csv file, with subject ID in the leftmost column.'
    )
    p.add_argument('-l', action='store', metavar='sep', dest='delimiter',
                    type=str, required=False, default=',',
                    help='The delimiter for the cohort file. Default is ",".'
    )
    p.add_argument('-F', '--filter', action='store', metavar='expr', dest='filters',
                    type=str, required=False, default=[], nargs="*",
                    help='Filters to apply to the cohort before analysis, e.g. Sex=Male, or Age>25, or Group=Group1,Group2 to eliminate Group3'
    )
    p.add_argument('--anderson-darling', action='store_true', dest='anderson_darling',
                    required=False, default=False, 
                    help='Perform the Anderson Darling test of normality per voxel'
    )
    p.add_argument('--mann-whitney', action='store_true', dest='mannwhitneyu',
                    required=False, default=False, 
                    help='Perform the Mann-Whitney U test of normality per voxel'
    )
    p.add_argument('-g', '--column', action='store', metavar='column', dest='group_column',
                    type=str, required=False, default='',
                    help='For two sample tests (Mann-Whitney U), the group column name in the header.'
    )
    p.add_argument('-a', '--groupA', action='store', metavar='groupA', dest='groupA',
                    type=str, required=False, default='',
                    help='The group label of the first group. If this group is higher, U value will be positive.'
    )
    p.add_argument('-b', '--groupB', action='store', metavar='groupB', dest='groupB',
                    type=str, required=False, default='',
                    help='The group label of the second group. If this group is higher, U value will be negative.'
    )
    return p

def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args()
    output_dir = os.path.dirname(os.path.realpath(args.outbase))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        if args.mannwhitneyu and ( not args.group_column or not args.groupA or not args.groupB ):
            parser.error("For Mann Whitney U test, -g column -a groupA and -b groupB must be defined")
        if not args.mannwhitneyu and not args.anderson_darling:
            parser.error("Nothing to do! Must provide at least one of --anderson-darling, --mann-whitney")
        logging.info("Read the cohort file")
        cohort = pd.read_csv(args.cohort, delimiter=args.delimiter)
        index_label = cohort.columns[0]
        cohort = cohort.set_index(index_label)
        subjects = list(cohort.index)
        logging.debug("Shape of cohort: {}".format(cohort.shape))
        for expr in args.filters:
            logging.info("Filter the cohort by {}".format(expr))
            cohort = filter_cohort(cohort, expr)
            logging.debug("Shape of cohort: {}".format(cohort.shape))
        filenames = [ args.filename_template.format(s=s) for s in subjects ]
        logging.info("Check existence of all input files")
        check_inputs(*filenames, nifti=True)
        # save cohort to outbase_cohort.csv 
        cohort.to_csv('{}_{}.csv'.format(args.outbase, 'cohort'), index=True, index_label=index_label)
        logging.info("Load Nifti mask and all data")
        mask_im = nib.load(args.mask)
        mask_data = mask_im.get_fdata() > 0
        data = asarray([ nib.load(args.filename_template.format(s=s)).get_fdata()[mask_data] for s in subjects ])
        if args.anderson_darling:
            logging.info("Perform voxelwise Anderson Darling test of normality")
            andersondarling_results = elementwise_anderson_darling(data)
            logging.info("Save results as Nifti")
            nifti_results = results_to_niftis(andersondarling_results, mask_im)
            for key in nifti_results.keys():
                nifti_results[key].to_filename('{0}_{1}.nii.gz'.format(args.outbase, key))
        if args.mannwhitneyu:
            logging.info("Perform voxelwise Mann Whitney U test")
            if not args.group_column in cohort.columns:
                raise ValueError("Specified group column not found in header of cohort file")
            subjects_groupA = list(cohort[cohort[args.group_column] == args.groupA].index)
            subjects_groupB = list(cohort[cohort[args.group_column] == args.groupB].index)
            if len(subjects_groupA) == 0:
                raise ValueError("No subjects found in groupA")
            if len(subjects_groupB) == 0:
                raise ValueError("No subjects found in groupB")
            filenames = [ args.filename_template.format(s=s) for s in subjects_groupA + subjects_groupB ]
            groupA_data = asarray([ nib.load(args.filename_template.format(s=s)).get_fdata()[mask_data] for s in subjects_groupA ])
            groupB_data = asarray([ nib.load(args.filename_template.format(s=s)).get_fdata()[mask_data] for s in subjects_groupB ])
            mannwhitneyu_results = elementwise_mannwhitneyu(groupA_data, groupB_data)
            logging.info("Save results as Nifti")
            nifti_results = results_to_niftis(mannwhitneyu_results, mask_im)
            for key in nifti_results.keys():
                nifti_results[key].to_filename('{0}_{1}.nii.gz'.format(args.outbase, key))
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise


if __name__ == '__main__':
    main(sys.argv[1:])
