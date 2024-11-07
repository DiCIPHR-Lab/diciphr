#! /usr/bin/env python

import os, sys, shutil, logging, argparse, traceback
from ..statistics.zscores import corrected_zscores
from ..statistics.utils import filter_cohort
import numpy as np
import pandas as pd 

DESCRIPTION = '''
    Calculates z-scores from data features including correction for covariates of interest.
'''

PROTOCOL_NAME='zscores'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    
    p.add_argument('-i', '--input', action='store', metavar='<csv>', dest='inputcsv',
                    type=str, required=True, 
                    help='The csv file containing sample features (e.g. roi stats) with subjectID in the leftmost column and, optionally, covariate columns'
                    )
    p.add_argument('-o', '--output', action='store', metavar='<csv>', dest='outputcsv',
                    type=str, required=True, 
                    help='The csv file to write the results'
                    )
    p.add_argument('-c', '--covars', action='store', metavar='<csv>', dest='covarscsv',
                    type=str, required=False, default=None,
                    help='A csv file containing covariate features (e.g. Age, Group, Site) with subjectID in the leftmost column, if input csv does not contain these columns'
                    )
    p.add_argument('-L', '--features', action='store', metavar='<txt>', dest='columnstxt',
                    type=str, required=False, default=None,
                    help='A txt file of column names to adjust. If not provided, will correct all columns that are detected as containing numerical data'
                    )
    p.add_argument('-f', '--formula', action='store', metavar='<str>', dest='formula',
                    type=str, required=False, default=None,
                    help='A patsy-like formula that defines the covariates to correct for. Default is no covariates. Example: Age+Sex'
                    )    
    p.add_argument('-F','--filter', action='store',metavar='str',dest='filters',
                    type=str, required=False, default=[], nargs="*", 
                    help='Filter(s) to apply which define the subset of cohort over which mean and SD for z-scores are calculated, e.g. Group=Control'
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

def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    dataframe = pd.read_csv(args.inputcsv, index_col=0, na_values=[' ','na','nan','NaN','NAN','NA','#N/A','.','NULL'])
    covars = None
    if args.covarscsv is not None:
        covars = pd.read_csv(args.covarscsv, index_col=0,na_values=[' ','na','nan','NaN','NAN','NA','#N/A','.','NULL'])
    columns = None
    if args.columnstxt is not None:
        columns = [r.strip() for r in open(args.columnstxt,'r').readlines()]
    if args.filters:
        subdf = covars.join(dataframe, how='inner')
        for fil in args.filters:
            subdf = filter_cohort(subdf, fil)
        subset = list(subdf.index)
        zscores_df = corrected_zscores(dataframe, subset=subset, columns=columns, covars=covars, formula=args.formula)
    else:
        zscores_df = corrected_zscores(dataframe, columns=columns, covars=covars, formula=args.formula)
    zscores_df.to_csv(args.outputcsv)
    
if __name__ == '__main__':
    main(sys.argv[1:])
