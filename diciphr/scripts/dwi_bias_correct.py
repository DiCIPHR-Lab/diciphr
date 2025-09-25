#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
                        DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import read_nifti, read_dwi, write_dwi 
from diciphr.diffusion import ( n4_bias_correct_dwi, 
                round_bvals, extract_b0, bet2_mask_nifti )
import nibabel as nib

DESCRIPTION = '''
    Bias Correct a DWI image 
'''

PROTOCOL_NAME='DWI_Bias_Correct'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwi',dest='dwi_file',
                    type=str, required=True, 
                    help='The DWI filename in Nifti format.'
                    )
    p.add_argument('-o',action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='Output filebase. Directory will be created if it does not exist.'
                    )
    p.add_argument('-m',action='store',metavar='mask',dest='mask_file',
                    type=str, required=False, default=None, 
                    help='The mask image, if not given will run bet2 on B0'
                    )
    p.add_argument('-b',action='store',metavar='bval',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r',action='store',metavar='bvec',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-i',action='store',metavar='int',dest='bias_iterations',
                    type=int, required=False, default=[50,50,50,50], nargs="*", 
                    help='Bias iterations. Default 50 50 50 50.'
                    )
    p.add_argument('-t',action='store',metavar='float',dest='bias_threshold', 
                    type=float, required=False, default=0.001,
                    help='Bias threshold. Default 0.001.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.dwi_file, nifti=True)
        check_inputs(output_dir, directory=True)
        if args.mask_file:
            check_inputs(args.mask_file, nifti=True)
        if args.bval_file:
            check_inputs(args.bval_file)
        if args.bvec_file:
            check_inputs(args.bvec_file)
        run_dwi_bias_correct(args.dwi_file, args.output, mask_file=args.mask_file, 
                bval_file=args.bval_file, bvec_file=args.bvec_file, 
                bias_iterations=args.bias_iterations, bias_threshold=args.bias_threshold)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_dwi_bias_correct(dwi_file, output, mask_file=None, bval_file=None, bvec_file=None, 
                bias_iterations=[50,50,50,50], bias_threshold=0.001):
    ''' 
    Run the pipeline.    
    '''
    logging.info('dwi_file: {}'.format(dwi_file))
    logging.info('output: {}'.format(output))
    if mask_file:
        logging.info('mask_file: {}'.format(mask_file))
    if bval_file:
        logging.info('bval_file: {}'.format(bval_file))
    if bvec_file:
        logging.info('bvec_file: {}'.format(bvec_file))
    
    logging.info('Begin {}'.format(PROTOCOL_NAME))    
    # Load dwi_file
    logging.info('Read input DWI')
    dwi_im, bvals, bvecs = read_dwi(dwi_file, bval_file=bval_file, bvec_file=bvec_file)
    bvals = round_bvals(bvals)
    
    if mask_file:
        logging.info('Read input mask')
        mask_im = read_nifti(mask_file)
    else:
        logging.info('Mask B0')
        b0_im = extract_b0(dwi_im, bvals)
        mask_im = bet2_mask_nifti(b0_im)
        
    dwi_bias_im, bias_im = n4_bias_correct_dwi(dwi_im, bvals, bvecs, mask_im, 
        iterations=bias_iterations, threshold=bias_threshold, return_field=True)
    write_dwi(output, dwi_bias_im, bvals, bvecs )
    
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
