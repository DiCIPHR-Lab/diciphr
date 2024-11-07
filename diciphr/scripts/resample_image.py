#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from ..utils import ( check_inputs, make_dir, is_writable,
                protocol_logging, DiciphrException )
from ..nifti_utils import ( read_nifti, read_dwi, write_nifti, write_dwi, 
                resample_image, strip_nifti_ext )
import nibabel as nib

DESCRIPTION = '''
    Resample a Nifti image.
'''

PROTOCOL_NAME='resample_image'   

allowed_interps = ['Linear','NearestNeighbor','MultiLabel','Gaussian','BSpline','GenericLabel',
        'CosineWindowedSinc','WelchWindowedSinc','HammingWindowedSinc','LanczosWindowedSinc'] 
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
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
    input = check_inputs(args.input, nifti=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    logging.info("Input: {}".format(args.input))
    logging.info("Output: {}".format(args.output))
    master_im = None
    if args.master_fn:
        logging.info("Master grid: {}".format(args.master_fn))
        master_im = read_nifti(args.master_fn)
    elif args.voxelsizes:
        if len(args.voxelsizes) == 1:
            args.voxelsizes *= 3
        if len(args.voxelsizes) not in (1,3):
            raise DiciphrException("Voxel sizes must be provided as 1 number or 3 numbers")
        logging.info("Voxel sizes: {}".format(args.voxelsizes))
    else:
        raise DiciphrException("One of voxel sizes or master must be provided")
    logging.info("Interpolation: {}".format(args.interp))
    run_resample_nifti(input, args.output, voxelsizes=args.voxelsizes, interp=args.interp, master_im=master_im)
    
def run_resample_nifti(input, output, voxelsizes=None, interp='Linear', master_im=None):
    outdir = os.path.dirname(os.path.realpath(output))
    if not os.path.isdir(outdir):
        raise DiciphrException('Output directory does not exist')
    if not is_writable(outdir):
        raise DiciphrException('Output directory is not writable')
    
    try:
        logging.info('Loading input.')
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
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
if __name__ == '__main__':
    main(sys.argv[1:])