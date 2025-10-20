#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti, write_nifti, read_dwi, synthstrip_mask_nifti
from diciphr.diffusion import extract_b0, round_bvals

DESCRIPTION = '''
    Runs SynthStrip on scalar images or on B0 images extracted from Diffusion MRI images
'''

PROTOCOL_NAME='SynthStrip'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', '--input', action='store', metavar='<nii>', dest='input',
                    type=str, required=True, 
                    help='Input Nifti filename'
                    )
    p.add_argument('-o', '--output', action='store', metavar='<path>', dest='output',
                    type=str, required=True, 
                    help='Output prefix for files. Skull-stripped image will be appended with "_brain.nii.gz", and mask image will be appended with "_mask.nii.gz"'
                    )
    p.add_argument('-S', '--sif', action='store', metavar='<sif>', dest='sif_file',
                    type=str, required=False, default=None, 
                    help='Path to SIF container. If not provided, will default to environmental variable DICIPHR_SYNTHSTRIPSIF'
                    )
    p.add_argument('-b', '--border', action='store', metavar='<int>', dest='border',
                    type=float, required=False, default=1, 
                    help='Border parameter. Smaller values (such as -1) will result in a tighter mask. Use higher values for bias correction mask. Default=1'
                    )
    p.add_argument('-f', '--fill', action='store', metavar='<int>', dest='fill',
                    type=int, required=False, default=1, 
                    help='Fill value. Sets the background voxel value. Default=0'
                    )
    p.add_argument('--no-csf', action='store_true', dest='no_csf',
                    help='Exclude CSF from brain border'
                    )
    p.add_argument('-d', '--distance', action='store_true', dest='distance',
                    help='Distance image will also be saved'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    outdir = os.path.dirname(os.path.realpath(args.output))
    make_dir(outdir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.input, nifti=True)
        try:
            dwi_im, bvals, bvecs = read_dwi(args.input)
            logging.info('Diffusion input detected')
            _diffusion = True
        except:
            logging.info('Non-diffusion input detected')
            input_im = read_nifti(args.input)
            _diffusion = False
        if _diffusion:
            logging.info("Extract B0 from diffusion image")
            input_im = extract_b0(dwi_im, round_bvals(bvals))
        logging.info("Run synthstrip")
        output_im, mask_im, distance_im = synthstrip_mask_nifti(input_im, sif_file=args.sif_file, 
                                   return_brain=True, return_distance=True, 
                                   border=args.border, fill=args.fill, no_csf=args.no_csf)
        if _diffusion:
            write_nifti(f'{args.output}_B0.nii.gz', output_im)
        else:
            write_nifti(f'{args.output}_brain.nii.gz', output_im)
        write_nifti(f'{args.output}_mask.nii.gz', mask_im)
        if args.distance:
            write_nifti(f'{args.output}_distance.nii.gz', distance_im)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
        
if __name__ == '__main__': 
    main(sys.argv[1:])
