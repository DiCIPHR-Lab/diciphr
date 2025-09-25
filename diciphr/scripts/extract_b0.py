#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging,
                    DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import read_nifti, read_dwi, multiply_images, has_nifti_ext
from diciphr.diffusion import round_bvals, extract_b0, bet2_mask_nifti, extract_shells_from_multishell_dwi
import nibabel as nib

DESCRIPTION = '''
    Extract the B0 from a DWI volume, by averaging. 
'''

PROTOCOL_NAME='Extract_B0'
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwi_file',dest='dwi_file',
                    type=str, required=True, 
                    help='The DWI filename in Nifti format.'
                    )
    p.add_argument('-o',action='store',metavar='output_base',dest='output_base',
                    type=str, required=True, 
                    help='Output filename. If not a nifti extension will append "_B0.nii.gz".'
                    )
    p.add_argument('-b',action='store',metavar='bval_file',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r',action='store',metavar='bvec_file',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('--bet', action='store_true', dest='run_bet', 
                    required=False, default=False, 
                    help='Skull strip the B0 with BET'
                    )
    p.add_argument('--noavg', action='store_true', dest='no_average', 
                    required=False, default=False, 
                    help='Do not average the B0, return a 4D Nifti.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    check_inputs(args.dwi_file, nifti=True)
    check_inputs(output_dir, directory=True)
    if args.bval_file:
        check_inputs(args.bval_file)
    if args.bvec_file:
        check_inputs(args.bvec_file)
    try:    
        run_extract_b0(args.dwi_file, args.output_base, 
            bval_file=args.bval_file, bvec_file=args.bvec_file, 
            run_bet=args.run_bet,
            no_average=args.no_average)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_extract_b0(dwi_file, output_base, bval_file=None, bvec_file=None, run_bet=False, no_average=False):
    ''' 
    Estimate a dti tensor
    
    Parameters
    ----------
    dwi_file : str
        Probtrackx directory.
    output_base : str
        Output file base
    run_bet : bool
        If true, skull strip B0 with BET
    Returns
    -------
    None
    '''
    logging.info('dwi_file: {}'.format(dwi_file))
    logging.info('output_base: {}'.format(output_base))
    if bval_file:
        logging.info('bval_file: {}'.format(bval_file))
    if bvec_file:
        logging.info('bvec_file: {}'.format(bvec_file))
    
    logging.info('Begin {}'.format(PROTOCOL_NAME))    
    logging.info('Read input DWI')
    dwi_im, bvals, bvecs = read_dwi(dwi_file, bval_file=bval_file, bvec_file=bvec_file)
    bvals = round_bvals(bvals)
    
    logging.info('Extract B0')
    if no_average and ((bvals==0).sum() > 1):
        b0_im, __, __ = extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, [0])
    else:
        b0_im = extract_b0(dwi_im, bvals)
        if run_bet:
            logging.info('Running BET on B0')
            mask_im = bet2_mask_nifti(b0_im)
            b0_im = multiply_images(b0_im, mask_im)
    if has_nifti_ext(output_base):
        output_b0_file = output_base
    else:
        output_b0_file = output_base+'_B0.nii.gz'
    b0_im.to_filename(output_b0_file)
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
