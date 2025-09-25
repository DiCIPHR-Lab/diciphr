#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_dwi, write_dwi
from diciphr.diffusion import concatenate_dwis, round_bvals

DESCRIPTION = '''
    Concatenate multiple DWI images to one image. 
'''

PROTOCOL_NAME='Concatenate_DWIs'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
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
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_filename))
    make_dir(output_dir,recursive=True,pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(*args.dwi_filenames, nifti=True)
        dwis = [ read_dwi(fn) for fn in args.dwi_filenames ]
        dwi, bvals, bvecs = concatenate_dwis(*dwis)
        if args.round:
            bvals = round_bvals(bvals)
        write_dwi(args.output_filename, dwi, bvals, bvecs)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

if __name__ == '__main__':
    main(sys.argv[1:])