#! /usr/bin/env python

import sys, argparse
from diciphr.nifti_utils import match_nifti_filenames 

DESCRIPTION = '''
    Parse a Nifti directory for images matching provided strings, 
'''

PROTOCOL_NAME='parse_nifti_directory'   

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', action='store',dest='directory',
                    type=str, required=True, 
                    help='The data directory.'
                    )
    p.add_argument('-m', action='store', dest='match_strings',
                    type=str, nargs="*", required=True, 
                    help='Strings used to match desired nifti files.'
                    )
    p.add_argument('-D', action='store_true', dest='diffusion',
                    required=False, 
                    help='If provided, will also search for bval/bvec files.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    res = match_nifti_filenames(args.directory, args.match_strings, json=True, diffusion=args.diffusion)
    for k in res.keys():
        print(k)
        formatted_filenames = []
        for f in res[k]:
            formatted_filenames.append('\''+f+'\'')
        print(' '.join(formatted_filenames))
    
if __name__ == '__main__':
    main(sys.argv[1:])