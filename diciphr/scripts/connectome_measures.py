#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
                    DiciphrArgumentParser, DiciphrException )
from diciphr.connectivity.connmat_utils import read_connmat, density, nodestrength, degree, prune_mat
from diciphr.connectivity.topology import ( efficiency_bin, efficiency_wei,
                    betweenness_bin, betweenness_wei, 
                    assortativity_bin, assortativity_wei,
                    transitivity_bin, transitivity_wei,
                    pathlength_wei, pathlength_bin,
                    modularity_louvain_wei, modularity_louvain_bin )
import numpy as np
import pandas as pd
from glob import glob 

DESCRIPTION = '''
    Calculate the BCT measures on a cohort of connectivity matrices.
'''

PROTOCOL_NAME='Connectome_Measures'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-m', action='store', metavar='mat', dest='matfile',
                    type=str, required=True, 
                    help='The connmat filenames. Either a direct path or a template with {s} in the place of the subject ID or matching string.'
                    )
    p.add_argument('-o', action='store', metavar='output', dest='output_base',
                    type=str, required=True, help='The csv output base name, will be appended with "_{density}.csv"'
                    )
    p.add_argument('-c', action='store', metavar='cohort', dest='cohort', 
                    type=str, required=False,
                    help='A cohort csv file or a list of subject IDs'
                    )
    p.add_argument('-C', '--header', action='store', metavar='header', dest='header',
                    type=str, required=False, 
                    help='If provided, this is the column name to find the subject ID or matching string. If not provided, assume a 1D list of strings with no header name'
                    )
    p.add_argument('-d', '--density', action='store', metavar='density', dest='densities', 
                    type=float, required=False, default=[100], nargs="*", 
                    help='A list of target densities, between 0 and 100, separated by spaces, to prune the connectomes to before computing stats. Default is [100] which will not prune the mats.'
                    )
    p.add_argument('-L', '--local', action='store_true', dest='local_only', 
                    required=False, default=False, help='If provided, compute only the local measures.'
                    )
    p.add_argument('-G', '--global', action='store_true', dest='global_only',
                    required=False, default=False, help='If provided, compute only the global measures.'
                    )
    p.add_argument('-B', '--binary', action='store_true', dest='binary_only', 
                    required=False, default=False, help='If provided, compute only the binary measures.'
                    )
    p.add_argument('-W', '--weighted', action='store_true', dest='weighted_only',
                    required=False, default=False, help='If provided, compute only the weighted measures.'
                    )
    p.add_argument('-n', '--nodes', action='store_true', dest='nodes', 
                    required=False, default=None, help='A text file containing the list of node names.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        if args.cohort:
            if args.header:
                subjects = list(pd.read_csv(args.cohort)[args.header])
            else:
                subjects = [ a.strip() for a in open(args.cohort, 'r').readlines() ]
            matfiles = [ args.matfile.format(s=subject) for subject in subjects ]
            matfiles = list(filter(lambda m: os.path.exists(m), matfiles))
            matfiles, subjects = zip(*list(filter(lambda m: os.path.exists(m[0]), zip(matfiles, subjects))))
        else:
            matfiles = sorted(glob(args.matfile.replace('{s}','*')))
            subjects = [ os.path.basename(m) for m in matfiles ]
        
        check_inputs(*matfiles)
        n = read_connmat(matfiles[0]).shape[0]
        if args.nodes:
            node_names = [ a.strip() for a in open(args.nodes, 'r').readlines() ][:n]
        else:
            node_names = None
        logging.info('Running connmat measures for {} subjects and {} densities'.format(len(subjects), len(args.densities)))
        dfs = connmat_measures(matfiles, subjects, args.densities, 
                    node_names=node_names, 
                    global_=not(args.local_only), local_=not(args.global_only),
                    binary=not(args.weighted_only), weighted=not(args.binary_only),
                    )
        for d in dfs:
            df = dfs[d]
            out_f = args.output_base+'_{}.csv'.format(d)
            logging.info('Writing sheet to filename {}'.format(out_f))
            df.to_csv(out_f, sep=',', index=True, header=True, index_label='Subject')
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def connmat_measures(matfiles, subjects, densities, node_names=None, global_=True, local_=True, binary=True, weighted=True):
    dfs = dict([ (d, pd.DataFrame(index=subjects)) for d in densities ])
    mats = [ read_connmat(m) for m in matfiles ]
    if node_names is None:
        node_names = ['node{}'.format(i) for i in range(len(mats[0]))]
    for d in densities:
        logging.info('Prune matrices to target density {}'.format(d))
        pruned_mats = [ prune_mat(m, density_target=d) for m in mats ] 
        nsubjs = len(subjects)
        for i, (s, m) in enumerate(zip(subjects, pruned_mats)):
            df = dfs[d]
            df.loc[s, 'density'] = density(m)
            if global_:
                logging.info('Global measures: {} out of {}'.format(i+1, nsubjs))
                # efficiency, pathlength, assortativity, transitivity, modularity
                if weighted:
                    df.loc[s, 'Global-efficiency-wtd'] = efficiency_wei(m, local=False)
                    df.loc[s, 'Global-charpath-wtd'] = pathlength_wei(m)
                    df.loc[s, 'Global-assortativity-wtd'] = assortativity_wei(m)
                    df.loc[s, 'Global-transitivity-wtd'] = transitivity_wei(m)
                    df.loc[s, 'Global-modularity-wtd'] = modularity_louvain_wei(m)
                if binary:
                    df.loc[s, 'Global-efficiency-bin'] = efficiency_bin(m, local=False)
                    df.loc[s, 'Global-charpath-bin'] = pathlength_bin(m)
                    df.loc[s, 'Global-assortativity-bin'] = assortativity_bin(m)
                    df.loc[s, 'Global-transitivity-bin'] = transitivity_bin(m)
                    df.loc[s, 'Global-modularity-bin'] = modularity_louvain_bin(m)
            if local_:
                logging.info('Local measures: {} out of {}'.format(i+1, nsubjs))
                # degree, nodestrength, efficiency, betweenness
                if weighted:
                    for n, val in zip(node_names, nodestrength(m)):
                        df.loc[s, 'Local-nodestrength-wtd_{}'.format(n)] = val
                    for n, val in zip(node_names, efficiency_wei(m, local=True)):
                        df.loc[s, 'Local-efficiency-wtd_{}'.format(n)] = val
                    for n, val in zip(node_names, betweenness_wei(m)):
                        df.loc[s, 'Local-betweenness-wtd_{}'.format(n)] = val
                if binary:
                    for n, val in zip(node_names, degree(m)):
                        df.loc[s, 'Local-degree-bin_{}'.format(n)] = val
                    for n, val in zip(node_names, efficiency_bin(m, local=True)):
                        df.loc[s, 'Local-efficiency-bin_{}'.format(n)] = val
                    for n, val in zip(node_names, betweenness_bin(m)):
                        df.loc[s, 'Local-betweenness-bin_{}'.format(n)] = val
    return dfs 
    
if __name__ == '__main__': 
    main(sys.argv[1:])
