#! /usr/bin/env python

import os, sys, logging, time
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti, read_dwi, json_files_from_niftis, split_image
from diciphr.diffusion import ( concatenate_dwis, round_bvals, extract_b0, 
               bet2_mask_nifti, most_gradients_pe, apply_topup, 
               prepare_acqparams_json, prepare_acqparams_nojson, 
               fsl_eddy, fsl_eddy_post_topup, save_eddy_text )

DESCRIPTION = '''
    Performs eddy on one or more DWI images.
    Multiple DWI images will be concatenated.
'''

PROTOCOL_NAME='DTI_Eddy'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-s', '--subject', action='store', metavar='<str>', dest='subject',
                    type=str, required=True,
                    help='The output basename / subject ID. Output files will be prepended with this string'
                    )
    p.add_argument('-d', '--dwi', action='store', metavar='<dwi>', dest='dwi_filenames',
                    type=str, required=True, nargs="*",
                    help='Path(s) of the DWI image files. Separate by spaces if multiple files. Associated bval/bvec files must exist'
                    )
    p.add_argument('-o', '--outdir', action='store', metavar='<dir>', dest='output_dir',
                    type=str, required=True, 
                    help='Name of output directory.'
                    )
    p.add_argument('-m', '--mask', action='store', metavar='<mask>', dest='mask', 
                    type=str, required=False, default=None, 
                    help='Provide a brain mask nifti file instead of other method(s)'
                    ) 
    p.add_argument('-t', '--topup', action='store', metavar='<topup>', dest='topup',
                    type=str, required=False,
                    help='The common prefix of results after running topup elsewhere.'
                    )
    p.add_argument('--index', action='store', metavar='<str>', dest='index',
                    type=str, required=False, default=None,
                    help='The index.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )
    p.add_argument('--acqparams', action='store', metavar='<str>', dest='acqparams',
                    type=str, required=False, default=None,
                    help='The acqparams.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )
    p.add_argument('--replace-outliers', action='store_true', dest='replace_outliers',
                    help='Use eddy method to replace outlier slices in the data.'
                    )
    p.add_argument('-T', '--readout-time', action='store', metavar='<float>', dest='readout_time',
                    type=float, required=False, default=0.062,
                    help='Total readout time (see FSL eddy documentation). Will be overridden by json file'
                    )
    p.add_argument('-P', '--phaseenc', action='store', metavar='<str>', dest='phase_encs',
                    type=str, required=False, nargs="*", default=[], 
                    help='The phase encoding direction(s) of the DWI image(s), provided in the same sequence as -d dwi inputs.'
                    )
    p.add_argument('-A', '--all-pes', action='store_true', dest='keep_all_pes',
                    help='If provided, will concatenate all phase encoding dirs into final DWI image. ' + 
                    'Use for data acquired with full sequences repeated with opposing phase encoding dirs. ' + 
                    'Default will keep the phase encoding direction with the most number of weighted volumes.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    make_dir(args.output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir or args.output_dir, 
                     filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(*args.dwi_filenames, nifti=True)
        if args.mask:
            check_inputs(args.mask, nifti=True)
        if args.bias_corr:
            args.bias_iterations = list(map(int, args.bias_iterations.split(',')))
        if args.bias_mask:
            check_inputs(args.bias_mask, nifti=True)
        if args.topup:
            try:
                check_inputs(args.topup, nifti=True)
                logging.info('Nifti reverse-phase-encoding-direction file found')
            except:
                # User provided topup prefix 
                check_inputs(args.topup+'_fieldcoef.nii.gz', args.topup+'_movpar.txt')
                if args.index:
                    check_inputs(args.index)
                elif not args.no_moco:
                    check_inputs(args.topup+'_index.txt')
                if args.acqparams:
                    check_inputs(args.acqparams)
                else:
                    check_inputs(args.topup+'_acqparams.txt')
                logging.info('Outputs of a previous run of FSL topup found')
        run_dti_eddy(args.subject, args.output_dir, args.dwi_filenames, mask=args.mask, 
                        topup=args.topup, index=args.index, acqparams=args.acqparams,
                        readout_time=args.readout_time, phase_encs=args.phase_encs, 
                        keep_all_pes=args.keep_all_pes, replace_outliers=args.replace_outliers)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def run_dti_eddy(subject, output_dir, dwi_filenames, mask=None, topup=None, 
                 index=None, acqparams=None, readout_time=0.062, phase_encs=[], 
                 keep_all_pes=False, replace_outliers=False):
    # log some info
    logging.info('subject: {}'.format(subject))
    logging.info('dwi_filenames: {}'.format(dwi_filenames))
    logging.info('output_dir: {}'.format(output_dir))
    if mask:
        logging.info('Mask: {}'.format(mask))
    if topup:
        logging.info('Topup: {}'.format(topup))
    
    # Output filenames
    eddy_text_prefix = os.path.join(output_dir, f"{subject}")   
    
    logging.info('Begin Protocol')
    
    # 1. Load dwi_filenames
    logging.info('Load dwi_filenames')
    dwi_ims = []
    bval_arrays = []
    bvec_arrays = []
    for dwifn in dwi_filenames:
        _d, _b, _v = read_dwi(dwifn, force=True)
        dwi_ims.append(_d)
        bval_arrays.append(round_bvals(_b))
        bvec_arrays.append(_v)
    if mask:
        logging.info('Load user provided mask')
        mask_im = read_nifti(mask)
    
    # 2. Get acquisition parameters from json files or from command line 
    json_files = json_files_from_niftis(dwi_filenames)
    if json_files:
        logging.info("Get acquisition parameters from .json files")
        all_acqparams = [prepare_acqparams_json(fn, dwi_im) for fn, dwi_im in zip(json_files, dwi_ims)]
    else:
        logging.info("Get acquisition parameters without .json files")
        if len(phase_encs) != len(dwi_filenames):
            if len(phase_encs) == 1:
                phase_encs = [phase_encs[0] for fn in dwi_filenames]
            elif len(phase_encs) > 1:
                raise ValueError("Number of phase encoding dirs does not match or could not be broadcast to number of DWI files")
        if len(phase_encs) > 0:
            logging.info("Get acquisition parameters without .json files")
        all_acqparams = [prepare_acqparams_nojson(readout_time, phase_enc) for phase_enc in phase_encs]
    # Array of which DWI images to keep in output 
    if keep_all_pes or len(all_acqparams) == 0:
        keep_dwis = [ True for dwi in dwi_filenames ]
    else:
        keep_dwis = most_gradients_pe(bval_arrays, all_acqparams)
    if not all(keep_dwis):
        main_phase_enc = [p for p,k in zip(phase_encs, keep_dwis) if k][0]
        topup_phase_enc = [p for p,k in zip(phase_encs, keep_dwis) if not k][0]
        logging.info(f"Processing diffusion images with predominant phase-encoding direction {main_phase_enc}")
        logging.info(f"Will use diffusion images with phase-encoding direction {topup_phase_enc} only for topup")
    else:
        logging.info("Processing all provided diffusion images")
        
    if topup:
        # Topup prefix was provided at command line and existence of files has been checked 
        logging.info("Using field estimate from previously run topup")
        if not acqparams:
            acqparams = topup+'_acqparams.txt'
        if not index:
            index = topup+'_index.txt'
        if os.path.exists(topup+'_b0u.nii.gz'):
            unwarped_b0_im = read_nifti(topup+'_b0u.nii.gz')
            if len(unwarped_b0_im.shape) == 4:
                unwarped_b0_im = split_image(unwarped_b0_im)[0]
        else:
            logging.info("Creating undistorted B0 image with applytopup on first B0")
            unwarped_b0_im = apply_topup(extract_b0(dwi_ims[0], bval_arrays[0], first=True), topup, acqparams)
       
    # Concatenate DWIs and extract shells 
    dwis = [(d,b,v) for d,b,v,k in zip(dwi_ims, bval_arrays, bvec_arrays, keep_dwis) if k]
    if len(dwis) > 1:
        dwi_proc_im, bvals, bvecs = concatenate_dwis(*dwis)
    else:
        dwi_proc_im, bvals, bvecs = dwis[0]
    
    # 7. Eddy 
    if topup:
        start_time = time.time()
        if mask is None:
            mask_im = bet2_mask_nifti(unwarped_b0_im, erode_iterations=0, f=0.2, g=0.0)
        # Run eddy 
        logging.info("Run Eddy")
        dwi_proc_im, bvals, bvecs, eddy_text_outputs = fsl_eddy_post_topup(
                    dwi_proc_im, bvals, bvecs, topup, acqparams, index, mask_im, 
                    unwarped_b0_im=unwarped_b0_im, replace_outliers=replace_outliers)                    
        save_eddy_text(eddy_text_prefix, eddy_text_outputs)
        end_time = time.time()
        logging.info("Done eddy. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    else:   
        start_time = time.time()
        if mask is None:
            b0_im = extract_b0(dwi_proc_im, bvals, first=True)
            mask_im = bet2_mask_nifti(b0_im, erode_iterations=0, f=0.2, g=0.0)
        logging.info("Run FSL eddy to correct for eddy and subject motion")
        dwi_proc_im, bvals, bvecs, eddy_text_outputs = fsl_eddy(
                    dwi_proc_im, bvals, bvecs, 
                    mask_im, 
                    readout_time=readout_time, 
                    replace_outliers=replace_outliers
        )
        save_eddy_text(eddy_text_prefix, eddy_text_outputs)
        end_time = time.time()
        logging.info("Done eddy. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
