#! /usr/bin/env python

import os, sys, logging, traceback, argparse, time
from ..utils import check_inputs, make_dir, protocol_logging
from ..nifti_utils import read_dwi, write_dwi
from ..diffusion import concatenate_dwis, round_bvals

DESCRIPTION = '''
    Concatenate multiple DWI images to one image. 
'''

PROTOCOL_NAME='Concatenate_DWIs'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', action='store', metavar='<dwi>', dest='dwi_filenames',
                    type=str, required=True, nargs="*",
                    help='Path(s) of the DWI image file. Separate by spaces if multiple files. Associated bval/bvec files must exist'
                    )
    p.add_argument('-o', action='store', metavar='<nii>', dest='output_filename',
                    type=str, required=True, 
                    help='Name of output DWI file. bval/bvec files will be written with corresponding basename.'
                    )
    p.add_argument('--round', action='store_true', dest='round', 
                    required=False, default=False, help='Round bvals to nearest hundred'
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
    output_dir = os.path.dirname(args.output_filename)
    make_dir(output_dir,recursive=True,pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(*args.dwi_filenames, nifti=True)
        dwis = [ read_dwi(fn) for fn in args.dwi_filenames ]
        dwi, bvals, bvecs = concatenate_dwis(*dwis)
        if args.round:
            bvals = round_bvals(bvals)
        write_dwi(args.output_filename, dwi, bvals, bvecs)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e

if __name__ == '__main__':
    main(sys.argv[1:])