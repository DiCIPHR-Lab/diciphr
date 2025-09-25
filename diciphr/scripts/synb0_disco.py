#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti, write_nifti, read_dwi, strip_nifti_ext
from diciphr.diffusion import ( synb0_disco, round_bvals, run_topup_post_synb0, 
                       prepare_acqparams_json, prepare_acqparams_nojson )

DESCRIPTION = '''
    Extract B0 and run SynB0-Disco on a DWI volume, then run topup configured for SynB0-Disco
'''

PROTOCOL_NAME='SynB0-Disco'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','--dwi', action='store',metavar='dwi_file',dest='dwi_file',
                    type=str, required=True, 
                    help='The DWI filename in Nifti format.'
                    )
    p.add_argument('-t','--t1', action='store',metavar='t1_file',dest='t1_file',
                    type=str, required=True, 
                    help='The T1 filename in Nifti format.'
                    )
    p.add_argument('-S','--sif', action='store',metavar='sif',dest='sif',
                    type=str, required=True, 
                    help='The SIF format image of the Synb0-Disco pipeline container'
                    )
    p.add_argument('-L','--license', action='store',metavar='license',dest='fslicense',
                    type=str, required=True, 
                    help='The Freesurfer license text file'
                    )
    p.add_argument('-o','--output', action='store',metavar='output_base',dest='output_base',
                    type=str, required=True, 
                    help='Output basename of undistorted synthetic B0 image and topup results.'
                    )
    p.add_argument('-m','--mask', action='store',metavar='mask_file',dest='mask_file',
                    type=str, required=False, default=None, 
                    help='The T1 mask image, if provided will skip Synb0-Disco pipeline skull stripping'
                    )
    p.add_argument('-b','--bvals', action='store',metavar='bval_file',dest='bval_file',
                    type=str, required=False, default=None,
                    help='The bvals file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-r','--bvecs', action='store',metavar='bvec_file',dest='bvec_file',
                    type=str, required=False, default=None,
                    help='The bvecs file, if not given will strip Nifti extension off the DWI image to ascertain filename.'
                    )
    p.add_argument('-T','--readout-time', action='store',metavar='float',dest='readout_time',
                    type=float, required=False, default=0.062, 
                    help='The readout time'
                    )        
    p.add_argument('-p','--phaseenc', action='store',metavar='pe_dir',dest='phase_enc',
                    type=str, required=False, default="AP",
                    help='The phase enconding direction. Either LR, RL, AP, PA, IS, SI'
                    )
    p.add_argument('-s','--smooth', action='store',metavar='<float>',dest='smooth_fwhm',
                    type=float, required=False, default=1.15,
                    help='Smooth the distorted B0 data before running topup. Set to 0 to disable smoothing. Default is 1.15.'
                    )
    p.add_argument('--topup', action='store_true', dest='topup', 
                    required=False, default=False, 
                    help='Run topup within SynB0-Disco container. Experimental'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.dwi_file, args.t1_file, nifti=True)
        check_inputs(args.sif, args.fslicense, nifti=False)
        check_inputs(output_dir, directory=True)
        if args.mask_file:
            check_inputs(args.mask_file, nifti=True)
        if args.bval_file:
            check_inputs(args.bval_file)
        if args.bvec_file:
            check_inputs(args.bvec_file)
        run_synb0_disco_and_topup(args.dwi_file, args.t1_file, args.output_base, args.sif, args.fslicense, 
                mask_file=args.mask_file, bval_file=args.bval_file, bvec_file=args.bvec_file,
                phase_enc=args.phase_enc, readout_time=args.readout_time, smooth_fwhm=args.smooth_fwhm,
                topup=args.topup)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_synb0_disco_and_topup(dwi_file, t1_file, output_base, sif, fslicense, 
            mask_file=None, bval_file=None, bvec_file=None, phase_enc='AP', readout_time=0.062,
            smooth_fwhm=1.15, topup=False):
    logging.info(f'DWI file: {dwi_file}')
    logging.info(f'T1 file: {t1_file}')
    logging.info(f'SIF file: {sif}')
    logging.info(f'Freesurfer license: {fslicense}')
    logging.info(f'Output base: {output_base}')
    if mask_file:
        logging.info(f'mask_file: {mask_file}')
    if bval_file:
        logging.info(f'bval_file: {bval_file}')
    if bvec_file:
        logging.info(f'bvec_file: {bvec_file}')
    logging.info(f'Begin {PROTOCOL_NAME}')  
    logging.info('Read input DWI')
    dwi_img, bvals, bvecs = read_dwi(dwi_file, bval_file=bval_file, bvec_file=bvec_file)
    logging.info('Read input T1')
    t1_img = read_nifti(t1_file)
    json_file = strip_nifti_ext(dwi_file)+'.json'
    if os.path.exists(json_file):
        acqparams_line = prepare_acqparams_json(json_file, dwi_img)
    else:
        acqparams_line = prepare_acqparams_nojson(readout_time, phase_enc)
    if mask_file:
        logging.info('Read input T1 mask')
        t1_mask_img = read_nifti(mask_file)
    else:
        t1_mask_img = None 
    bvals = round_bvals(bvals)
    logging.info('Run SynB0-Disco pipeline')
    synb0_output_file = f'{output_base}_synb0.nii.gz'
    if not os.path.exists(synb0_output_file):
        synb0_img = synb0_disco(sif, fslicense, dwi_img, bvals, acqparams_line, t1_img, t1_mask_img=t1_mask_img, topup=topup)
        logging.info('Write resulting undistorted B0 image')
        write_nifti(synb0_output_file, synb0_img)
    else:
        synb0_img = read_nifti(synb0_output_file)
    logging.info('Run topup with SynB0-Disco configuration file')
    run_topup_post_synb0(dwi_img, bvals, bvecs, synb0_img, acqparams_line, 
                output_base, smooth_fwhm=smooth_fwhm)
    logging.info('Done')
    
if __name__ == '__main__': 
    main(sys.argv[1:])
