#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from ..utils import ( check_inputs, make_dir, 
                protocol_logging, DiciphrException )
from ..nifti_utils import ( read_nifti, write_nifti, 
                read_dwi, write_dwi, is_valid_dwi )
from ..diffusion import round_bvals, extract_shells_from_multishell_dwi
import nibabel as nib

DESCRIPTION = '''
    Extract shells from a multishell DWI. 
'''

PROTOCOL_NAME='Extract_Shells_from_Multishell_DWI'    
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwi_file',dest='dwi_file',
                    type=str, required=True, 
                    help='Input DWI filename'
                    )
    p.add_argument('-o',action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='Output filename'
                    )
    p.add_argument('-s',action='store',metavar='list',dest='shells',
                    type=str, required=True, 
                    help='Shells to keep, separated by commas. B=0 will be kept.'
                    )
    p.add_argument('-b',action='store',metavar='bval_file',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r',action='store',metavar='bvec_file',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('--no-bzero',action='store_true',dest='no_bzero',
                    required=False, default=False, 
                    help='If provided, B=0 may be removed.'
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
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(args.dwi_file, nifti=True)
        if args.bval_file: 
            check_inputs(args.bval_file)
        if args.bvec_file: 
            check_inputs(args.bvec_file)
        check_inputs(output_dir, directory=True)
        shells = list(map(int,args.shells.split(',')))
        if args.no_bzero:
            logging.info('Will remove B=0 volumes')
        elif 0 not in shells:
            shells = [0] + shells
        run_extract_shells_from_multishell_dwi(args.dwi_file, args.output, shells, 
                bval_file=args.bval_file, bvec_file=args.bvec_file)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
def run_extract_shells_from_multishell_dwi(dwi_file, output, shells, bval_file=None, bvec_file=None):
    ''' 
    Remove DWI gradients 
    '''
    logging.info('DWI: {}'.format(dwi_file))
    if bval_file:
        logging.info('bvals: {}'.format(bval_file))
    if bvec_file:
        logging.info('bvecs: {}'.format(bvec_file))
    logging.info('Output file: {}'.format(output))
    logging.info('Shells: {}'.format(shells))
    
    logging.info('Begin Protocol {}'.format(PROTOCOL_NAME))    
    logging.info('Read input nifti')
    dwi_im, bvals, bvecs = read_dwi(dwi_file, bval_file, bvec_file)
    bvals = round_bvals(bvals)
    out_dwi_im, out_bvals, out_bvecs = extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, shells)
    if 0 in shells:
        is_valid_dwi(out_dwi_im, out_bvals, out_bvecs, True)
    write_dwi(output, out_dwi_im, out_bvals, out_bvecs)
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
