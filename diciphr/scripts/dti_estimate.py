#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrException, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti, read_dwi
from diciphr.diffusion import ( estimate_tensor, estimate_tensor_restore, TensorScalarCalculator,                 
                extract_shells_from_multishell_dwi, round_bvals, extract_b0, bet2_mask_nifti )
import nibabel as nib

DESCRIPTION = '''
    Estimate a tensor and calculate FA, Trace from a diffusion weighted volume
'''

PROTOCOL_NAME='DTI_Estimate'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwi_file',dest='dwi_file',
                    type=str, required=True, 
                    help='The DWI filename in Nifti format.'
                    )
    p.add_argument('-o',action='store',metavar='output_base',dest='output_base',
                    type=str, required=True, 
                    help='Output filebase. Tensor file will be written to outputbase_tensor.nii.gz. Directory will be created if it does not exist.'
                    )
    p.add_argument('-m',action='store',metavar='mask_file',dest='mask_file',
                    type=str, required=False, default=None, 
                    help='The mask image, if not given will run bet2 on B0'
                    )
    p.add_argument('-b',action='store',metavar='bval_file',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r',action='store',metavar='bvec_file',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-s','--extract-shell', action='store', metavar='<int>', dest='extract_shell', 
                    type=int, required=False, default=None, 
                    help='Extract the shell matching this b-value from multishell data.'
                    )
    p.add_argument('--fit', action='store', metavar='str', dest='fit_method', 
                    required=False, default='WLS', 
                    help='The fit method, one of WLS, OLS, NLLS. Default is WLS.'
                    )                
    p.add_argument('--restore', action='store_true', dest='restore', 
                    required=False, default=False, 
                    help='Use the RESTORE algorithm for tensor fit.'
                    )
    p.add_argument('-N', action='store', metavar='<int>', dest='N', 
                    type=int, required=False, default=0, 
                    help='N paramter for RESTORE algorithm. Use 1 for SENSE (Philips), number of coils for GRAPPA (GE, Siemens), or 0 for Gaussian (default).'
                    )
    p.add_argument('--erode', action='store_true', dest='erode', 
                    required=False, default=False, 
                    help='Erode the brain mask one time.' 
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
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
        run_dti_estimate(args.dwi_file, args.output_base, mask_file=args.mask_file, bval_file=args.bval_file, bvec_file=args.bvec_file, extract_shell=args.extract_shell, fit_method=args.fit_method, restore=args.restore, restore_N=args.N, erode=args.erode)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_dti_estimate(dwi_file, output_base, mask_file=None, bval_file=None, bvec_file=None, 
                extract_shell=None, fit_method='WLS', restore=False, restore_N=0, erode=False):
    ''' 
    Run the pipeline.    
    '''
    logging.info('dwi_file: {}'.format(dwi_file))
    logging.info('output_base: {}'.format(output_base))
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
    if extract_shell is not None:
        logging.info('Extract the {0} shell from multishell data.'.format(extract_shell))
        extract_shell = [0, extract_shell]        
        dwi_im, bvals, bvecs = extract_shells_from_multishell_dwi(dwi_im, bvals, bvecs, extract_shell)
    if mask_file:
        logging.info('Read input mask')
        mask_im = read_nifti(mask_file)
    else:
        logging.info('Mask B0')
        b0_im = extract_b0(dwi_im, bvals)
        if erode:
            erode_iterations = 1
        else:
            erode_iterations = 0
        mask_im = bet2_mask_nifti(b0_im, erode_iterations=erode_iterations)
    if restore:
        logging.info('Estimate tensor with RESTORE algorithm')
        tensor_im = estimate_tensor_restore(dwi_im, mask_im, bvals, bvecs, N=restore_N)
    else:
        logging.info('Estimate tensor')
        tensor_im = estimate_tensor(dwi_im, mask_im, bvals, bvecs, fit_method=fit_method)
    logging.info('Calculate DTI metrics')
    TSC = TensorScalarCalculator(tensor_im, mask_im)
    logging.info('Write results')
    tensor_filename = output_base+'_tensor.nii.gz'
    fa_filename = output_base+'_tensor_FA.nii.gz'
    tr_filename = output_base+'_tensor_TR.nii.gz'
    md_filename = output_base+'_tensor_MD.nii.gz'
    ax_filename = output_base+'_tensor_AX.nii.gz'
    rad_filename = output_base+'_tensor_RAD.nii.gz'
    mask_filename = output_base+'_tensor_mask.nii.gz'
    tensor_im.to_filename(tensor_filename)
    logging.info('Wrote {}'.format(tensor_filename))
    TSC.FA.to_filename(fa_filename)
    logging.info('Wrote {}'.format(fa_filename))
    TSC.TR.to_filename(tr_filename)
    logging.info('Wrote {}'.format(tr_filename))
    TSC.MD.to_filename(md_filename)
    logging.info('Wrote {}'.format(md_filename))
    TSC.AX.to_filename(ax_filename)
    logging.info('Wrote {}'.format(ax_filename))
    TSC.RAD.to_filename(rad_filename)
    logging.info('Wrote {}'.format(rad_filename))
    mask_im.to_filename(mask_filename)
    logging.info('Wrote {}'.format(mask_filename))    
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
