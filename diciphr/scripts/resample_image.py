#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, is_writable,
                protocol_logging, DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, read_dwi, write_nifti, write_dwi, 
                resample_image, strip_nifti_ext )
import nibabel as nib

DESCRIPTION = '''
    Resample a Nifti image.
'''

PROTOCOL_NAME='resample_image'   

allowed_interps = ['Linear','NearestNeighbor','MultiLabel','Gaussian','BSpline','GenericLabel',
        'CosineWindowedSinc','WelchWindowedSinc','HammingWindowedSinc','LanczosWindowedSinc'] 
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i',action='store',metavar='input',dest='input',
                    type=str, required=True, 
                    help='The input image. If a DWI, bvals and bvecs will be copied over.'
                    )
    p.add_argument('-o',action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='The output image. Output directory has to exist.'
                    )
    p.add_argument('-r',action='store',metavar='dim',dest='voxelsizes',
                    type=float, required=False, default=None, nargs='*', 
                    help='The target voxel sizes. Can be a single number for isotropic resampling or 3 values separated by spaces.'
                    )
    p.add_argument('-n',action='store',metavar='mode',dest='interp',
                    type=str, required=False, default='Linear',
                    help='The interpolation mode (case-sensitive). Choices are "{}"'.format('", "'.join(allowed_interps))
                    )
    p.add_argument('-m', '--master', action='store', metavar='nii', dest='master_fn', 
                    type=str, required=False, default=None,
                    help='Align dataset grid to that of this image')
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.input, nifti=True)
        logging.info("Input: {}".format(args.input))
        logging.info("Output: {}".format(args.output))
        
        run_resample_nifti(args.input, args.output, voxelsizes=args.voxelsizes, interp=args.interp, master_fn=args.master_fn)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_resample_nifti(input, output, voxelsizes=[], interp='Linear', master_fn=None):
    master_im = None 
    if master_fn:
        logging.info("Master grid: {}".format(master_fn))
        master_im = read_nifti(master_fn)
    elif voxelsizes:
        if len(voxelsizes) == 1:
            voxelsizes *= 3
        elif len(voxelsizes) != 3:
            raise DiciphrException("Voxel sizes must be provided as 1 number or 3 numbers")
        logging.info("Voxel sizes: {}".format(voxelsizes))
    else:
        raise DiciphrException("One of voxel sizes or master must be provided")
    logging.info("Interpolation: {}".format(interp))
    
    try:
        nifti_im, bvals, bvecs = read_dwi(input)
        logging.info('DWI input detected.')
        is_dwi = True
    except DiciphrException:
        logging.info('Non-DWI input detected.')
        nifti_im = read_nifti(input)
        is_dwi = False
    resampled_im = resample_image(nifti_im, voxelsizes, interp=interp, master=master_im)
    if is_dwi:
        write_dwi(output, resampled_im, bvals, bvecs)
        logging.info("Wrote DWI output to {0} {1} {2}".format(output, strip_nifti_ext(output)+'.bval', strip_nifti_ext(output)+'.bvec'))
    else:
        write_nifti(output, resampled_im)

if __name__ == '__main__':
    main(sys.argv[1:])