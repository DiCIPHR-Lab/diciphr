#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_dwi, write_dwi, is_valid_dwi
from diciphr.diffusion import round_bvals, extract_shells_from_multishell_dwi, extract_gaussian_shells

DESCRIPTION = '''
    Extract shells from a DWI. 
'''

PROTOCOL_NAME='Extract_Shells_from_DWI'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','--dwi', action='store',metavar='dwi_file',dest='dwi_file',
                    type=str, required=True, 
                    help='Input DWI filename'
                    )
    p.add_argument('-o','--output', action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='Output filename'
                    )
    p.add_argument('-s', '--shells', action='store',metavar='list',dest='shells',
                    type=str, required=False, default='', 
                    help='Shells to keep, separated by commas. B=0 will be kept.'
                    )
    p.add_argument('-g', '--gaussian', action='store_true', dest='gaussian', 
                    help='Extract the Gaussian (b-value 500-1500) shells from a DWI'
                    )
    p.add_argument('-b', '--bvals', action='store',metavar='bval_file',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r', '--bvecs', action='store',metavar='bvec_file',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('--no-bzero',action='store_true',dest='no_bzero',
                    required=False, default=False, 
                    help='If provided, B=0 will be removed.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        if not args.shells or args.gaussian:
            raise ValueError("One of --shells or --gaussian must be provided.")
        output_dir = os.path.dirname(os.path.realpath(args.output))
        make_dir(output_dir, recursive=True, pass_if_exists=True)
        check_inputs(args.dwi_file, nifti=True)
        if args.bval_file: 
            check_inputs(args.bval_file)
        if args.bvec_file: 
            check_inputs(args.bvec_file)
        check_inputs(output_dir, directory=True)
        run_extract_shells(args.dwi_file, args.output, shells=args.shells, gaussian=args.gaussian, 
                    no_bzero=args.no_bzero, bval_file=args.bval_file, bvec_file=args.bvec_file)       
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def run_extract_shells(dwi_file, output, shells='', gaussian=False, no_bzero=False, bval_file=None, bvec_file=None):
    logging.info('Begin Protocol {}'.format(PROTOCOL_NAME))    
    if shells:
        shells = list(map(int, shells.split(',')))
    else:
        shells = []
    logging.info('DWI: {}'.format(dwi_file))
    if bval_file:
        logging.info('bvals: {}'.format(bval_file))
    if bvec_file:
        logging.info('bvecs: {}'.format(bvec_file))
    logging.info('Output file: {}'.format(output))
    logging.info('Shells: {}'.format(shells))
    logging.info('Gaussian: {}'.format(gaussian))
    logging.info('B0 included in output: {}'.format(not no_bzero))
    logging.info('Read input DWI')
    dwi_im, bvals, bvecs = read_dwi(dwi_file, bval_file, bvec_file)
    bvals = round_bvals(bvals)
    if no_bzero:
        logging.info('Will remove B=0 volumes')
    elif shells and 0 not in shells:
        shells = [0] + shells
    if gaussian:
        dwi_im, bvals, bvecs = extract_gaussian_shells(dwi_im, bvals, bvecs)
    if shells:
        dwi_im, bvals, bvecs = extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, shells)
    if not no_bzero:
        is_valid_dwi(dwi_im, bvals, bvecs, True)
    write_dwi(output, dwi_im, bvals, bvecs)
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
