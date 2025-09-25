#! /usr/bin/env python

import os, sys, logging, time
from collections import OrderedDict
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
               DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, write_nifti, read_dwi, write_dwi, 
               json_files_from_niftis, resample_image, is_nifti_file, split_image )
from diciphr.diffusion import ( concatenate_dwis, round_bvals, extract_b0, 
               extract_shells_from_multishell_dwi, extract_gaussian_shells, 
               mppca_denoise, gibbs_unringing, n4_bias_correct_dwi, bet2_mask_nifti, 
               most_gradients_pe, prepare_acqparams_json, prepare_acqparams_nojson, 
               run_topup, apply_topup, fsl_eddy, fsl_eddy_post_topup, save_eddy_text, 
               estimate_tensor, TensorScalarCalculator )

DESCRIPTION = '''
    Performs preprocessing on one or more DWI images.
    Multiple DWI images will be concatenated.
'''

PROTOCOL_NAME='DTI_Preprocess'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    
    g_io = p.add_argument_group('Required Input/Output arguments')
    g_io.add_argument('-s', '--subject', action='store', metavar='<str>', dest='subject',
                    type=str, required=True,
                    help='The output basename / subject ID. Output files will be prepended with this string'
                    )
    g_io.add_argument('-d', '--dwi', action='store', metavar='<dwi>', dest='dwi_filenames',
                    type=str, required=True, nargs="*",
                    help='Path(s) of the DWI image files. Separate by spaces if multiple files. Associated bval/bvec files must exist'
                    )
    g_io.add_argument('-o', '--outdir', action='store', metavar='<dir>', dest='output_dir',
                    type=str, required=False, default=None,
                    help='Name of output directory. Default: ./{subject}/DTI_Preprocess'
                    )
    
    g_oio = p.add_argument_group('Optional Input/Output arguments')
    g_oio.add_argument('-m', '--mask', action='store', metavar='<mask>', dest='mask', 
                    type=str, required=False, default=None, 
                    help='Provide a brain mask nifti file instead of other method(s)'
                    ) 
    g_oio.add_argument('-t', '--topup', action='store', metavar='<topup>', dest='topup',
                    type=str, required=False,
                    help='The common prefix of results after running topup elsewhere.'
                    )
    g_oio.add_argument('--index', action='store', metavar='<str>', dest='index',
                    type=str, required=False, default=None,
                    help='The index.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )
    g_oio.add_argument('--acqparams', action='store', metavar='<str>', dest='acqparams',
                    type=str, required=False, default=None,
                    help='The acqparams.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )
    
    g_dn = p.add_argument_group('Denoising options')
    g_dn.add_argument('--no-denoise', action='store_false', dest='denoise',
                    help='Skip MPPCA denoising'
                    )
    g_dn.add_argument('-G', '--gibbs', action='store_true', dest='gibbs',
                    help='Run Gibbs unringing on the data (default: False)'
                    )
    g_dn.add_argument('--acquisition-slice', action='store', dest='acquisition_slicetype', 
                    required=False, default='axial', 
                    help='The acquisition slice, one of axial (default), sagittal, coronal. Used for Gibbs unringing.'
                    )
                    
    g_e = p.add_argument_group('Motion and distortion correction options')
    g_e.add_argument('--no-moco', action='store_true', dest='no_moco',
                    help='Skip FSL eddy'
                    ) 
    g_e.add_argument('--replace-outliers', action='store_true', dest='replace_outliers',
                    help='Use eddy method to replace outlier slices in the data.'
                    )
    g_e.add_argument('-T', '--readout-time', action='store', metavar='<float>', dest='readout_time',
                    type=float, required=False, default=0.062,
                    help='Total readout time (see FSL eddy documentation). Will be overridden by json file'
                    )
    g_e.add_argument('-P', '--phaseenc', action='store', metavar='<str>', dest='phase_encs',
                    type=str, required=False, nargs="*", default=[], 
                    help='The phase encoding direction(s) of the DWI image(s), provided in the same sequence as -d dwi inputs.'
                    )
    g_e.add_argument('--config', action='store', metavar='<nii>', dest='config', 
                    type=str, required=False, default=None,
                    help='A configuration file for FSL topup. Will default to Synb0 configuration parameters for Synb0-DISCO, or b02b0.cnf with reverse PE scan' 
                    )
    
    g_b = p.add_argument_group('Bias correction options')
    g_b.add_argument('--no-bias', action='store_false', dest='bias_corr', 
                    help='Skip bias correction of the DWI with ants N4BiasFieldCorrection'
                    )
    g_b.add_argument('--bias-mask', action='store', dest='bias_mask', metavar='<mask>',
                    type=str, required=False, default=None,
                    help='A mask to weight the bias correction, such as brain without tumor. Default is the brain mask.'
                    )                    
    g_b.add_argument('--bias-iterations', action='store', dest='bias_iterations', metavar='<int>',
                    type=str, required=False, default='50,50,50,50', 
                    help='The iterations to pass to N4BiasFieldCorrection separated by commas. Default is 50,50,50,50'
                    )                    
    g_b.add_argument('--bias-threshold', action='store', dest='bias_threshold', metavar='<float>',
                    type=float, required=False, default=0.001,
                    help='The threshold to pass to N4BiasFielsqdCorrection. Default is 0.001'
                    )  
    
    g_o = p.add_argument_group('Miscellaneous options')
    g_o.add_argument('-A', '--all-pes', action='store_true', dest='keep_all_pes',
                    help='If provided, will concatenate all phase encoding dirs into final DWI image. ' + 
                    'Use for data acquired with full sequences repeated with opposing phase encoding dirs. ' + 
                    'Default will keep the phase encoding direction with the most number of weighted volumes.'
                    )
    g_o.add_argument('-r', '--resample', action='store', metavar='<float>', dest='resample',
                    type=float, required=False, default=0, 
                    help='Resample DWI to an isotropic resolution. Default is 0, for no resampling.'
                    )
    g_o.add_argument('-x','--extract-shell', action='store', dest='extract_shell', metavar='bvalue', 
                    type=int, required=False, default=None, nargs="*",
                    help='Extract shell at this/these bvalue(s) before preprocessing'
                    )
    g_o.add_argument('--normalize', action='store_true', dest='normalize',
                    help='Normalize the DWI to reference B0 value of 1000.0 before processing.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), args.subject, 'DTI_Preprocess')
    make_dir(args.output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir or args.output_dir, 
                     filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        dwi_filenames = []
        for d in args.dwi_filenames:
            d = os.path.realpath(d)
            if os.path.isdir(d):
                dwi_filenames.extend(dwi_filenames_from_directory(d))
            elif os.path.exists(d):
                dwi_filenames.append(d)
            else:
                raise FileNotFoundError(f"Input path does not exist: {d}")
        check_inputs(*dwi_filenames, nifti=True)
        args.dwi_filenames = dwi_filenames
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
        run_dti_preprocess(args.subject, args.output_dir, args.dwi_filenames, mask=args.mask, 
                        topup=args.topup, config=args.config, index=args.index, acqparams=args.acqparams,
                        keep_all_pes=args.keep_all_pes, extract_shell=args.extract_shell, normalize=args.normalize, 
                        no_moco=args.no_moco, denoise=args.denoise, gibbs=args.gibbs,
                        acquisition_slicetype=args.acquisition_slicetype,
                        bias_corr=args.bias_corr, bias_mask=args.bias_mask,
                        bias_iterations=args.bias_iterations, bias_threshold=args.bias_threshold, 
                        resample=args.resample, phase_encs=args.phase_encs, readout_time=args.readout_time, 
                        replace_outliers=args.replace_outliers)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def group_by_phaseenc(dwi_images, phase_encs, readout_times):
    ret = OrderedDict()
    for i, (dwi, phase_enc, readout_time) in enumerate(zip(dwi_images, phase_encs, readout_times)):
        if phase_enc not in ret.keys():
            ret[phase_enc] = [] 
        ret[phase_enc].append((i, dwi, phase_enc, readout_time))
    return ret 

def unique_acqparams(acqparams_list):
    uniq = set()
    for acqparams_line in acqparams_list:
        acq = acqparams_line[:3]
        uniq.add(tuple(acq))
    return len(uniq)
  
def dwi_filenames_from_directory(directory):
    all_files = sorted(os.listdir(directory))
    bval_files = list(filter(lambda fn: fn.endswith('.bval'), all_files))
    if len(bval_files) == 0:
        raise DiciphrException('No diffusion .bval files found in directory')
    nifti_files = []
    for bv in bval_files:
        for ext in ['nii.gz', 'nii', 'hdr']:
            fn = os.path.join(directory, bv[:-4]+ext)
            logging.info(fn)
            if is_nifti_file(fn):
                nifti_files.append(fn)
                break 
            raise DiciphrException('Nifti file corresponding to bval file does not exist')
    return nifti_files
  
def run_dti_preprocess(subject, output_dir, dwi_filenames, json_filenames=[], bval_filenames=[], bvec_filenames=[], 
               mask=None, topup=None, config=None, acqparams=None, index=None, extract_shell=None, normalize=False, 
               no_moco=False, denoise=False, gibbs=False, acquisition_slicetype='axial', 
               replace_outliers=False, phase_encs=[], readout_time=0.062, keep_all_pes=False,
               bias_corr=True, bias_mask=None, bias_iterations=[50,50,50,50], bias_threshold=0.001, 
               resample=0, bet_f=0.2, bet_g=0.0):
    # log some info
    logging.info('subject: {}'.format(subject))
    logging.info('dwi_filenames: {}'.format(dwi_filenames))
    logging.info('output_dir: {}'.format(output_dir))
    if resample > 0: 
        logging.info('resample: {}'.format(resample))
    else:
        logging.info('Not resampling data.')
    if mask:
        logging.info('Mask: {}'.format(mask))
    if topup:
        logging.info('Topup: {}'.format(topup))
    if extract_shell:
        if 0 not in extract_shell:
            extract_shell = [0]+extract_shell
        logging.info('Extract shell: {}'.format(extract_shell))    
    if no_moco: 
        logging.info('Skipping motion correction step.')
    if not denoise:
        logging.info('Skipping denoising step.')
    
    # Output filenames
    dwi_processed_filename = os.path.join(output_dir, f"{subject}_DWI_preprocessed.nii.gz")
    # noise_filename = os.path.join(output_dir, f"{subject}_denoising_diff.nii.gz")
    mask_filename = os.path.join(output_dir, f"{subject}_tensor_mask.nii.gz")
    tensor_filename = os.path.join(output_dir, f"{subject}_tensor.nii.gz")
    fa_filename = os.path.join(output_dir, f"{subject}_tensor_FA.nii.gz")
    tr_filename = os.path.join(output_dir, f"{subject}_tensor_TR.nii.gz")        
    md_filename = os.path.join(output_dir, f"{subject}_tensor_MD.nii.gz")        
    ax_filename = os.path.join(output_dir, f"{subject}_tensor_AX.nii.gz")        
    rad_filename = os.path.join(output_dir, f"{subject}_tensor_RAD.nii.gz")        
    b0_filename = os.path.join(output_dir, f"{subject}_B0.nii.gz")    
    eddy_text_prefix = os.path.join(output_dir, f"{subject}")   
    bias_filename = os.path.join(output_dir, f"{subject}_bias_field.nii.gz")
    topup_base = os.path.join(output_dir, f"{subject}_topup")
    
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
    if bias_mask:
        logging.info('Load user provided bias weight mask')
        bias_mask_im = read_nifti(bias_mask)
    
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
        
    # 3. Topup before denoising
    if topup:
        # Topup prefix was provided at command line and existence of files has been checked 
        logging.info("Using field estimate from previously run topup")
        if not acqparams:
            acqparams = topup+'_acqparams.txt'
        if not index:
            index = topup+'_index.txt'
#        if os.path.exists(topup+'_b0u.nii.gz'):
#            unwarped_b0_im = read_nifti(topup+'_b0u.nii.gz')
#            if len(unwarped_b0_im.shape) == 4:
#                unwarped_b0_im = split_image(unwarped_b0_im)[0]
#        else:
#            logging.info("Creating undistorted B0 image with applytopup on first B0 to estimate a brain mask")
#            unwarped_b0_im = apply_topup(extract_b0(dwi_ims[0], bval_arrays[0], first=True), topup, acqparams)
        logging.info("Creating undistorted B0 image with applytopup on first B0 to estimate a brain mask")
        unwarped_b0_im = apply_topup(extract_b0(dwi_ims[0], bval_arrays[0], first=True), topup, acqparams)
    elif unique_acqparams(all_acqparams) > 1:
        # run topup
        logging.info("Run topup")
        topup = run_topup(dwi_ims, bval_arrays, bvec_arrays, all_acqparams, topup_base, keep_dwis=keep_dwis)
        acqparams = topup+'_acqparams.txt'
        index = topup+'_index.txt'
        unwarped_b0_im = read_nifti(topup+'_b0u.nii.gz')
        if len(unwarped_b0_im.shape) == 4:
            unwarped_b0_im = split_image(unwarped_b0_im)[0]
    elif unique_acqparams == 1 and len(dwi_ims) > 1:
        # When there is only one group of phase encoding directions, but multiple DWI images, concatenate them BEFORE denoising
        dwi_ims, bval_arrays, bvec_arrays = [[x] for x in concatenate_dwis(*zip(dwi_ims, bval_arrays, bvec_arrays))]
        keep_dwis = [True]
        
    # 4. Denoising
    if denoise:
        logging.info("Denoising DWI Volume(s) with MP-PCA")
        start_time = time.time()
        for i, (dwi_im, bvals, bvecs, keep) in enumerate(zip(dwi_ims, bval_arrays, bvec_arrays, keep_dwis)):
            if keep:
                dwi_im = mppca_denoise(dwi_im, bvals, bvecs, patch_radius=2, return_diff=False)
                dwi_ims[i] = dwi_im 
        end_time = time.time()
        logging.info("Done Denoising. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    else:
        logging.info("Skipping data denoising.")

    # 5. Gibbs Unringing
    if gibbs:
        logging.info("Gibbs unringing")
        start_time = time.time()
        for i, (dwi_im, bvals, bvecs, keep) in enumerate(zip(dwi_ims, bval_arrays, bvec_arrays, keep_dwis)):
            if keep:
                dwi_im = gibbs_unringing(dwi_im, acquisition_slicetype=acquisition_slicetype, n_points=3, num_processes=1, return_diff=False)
                dwi_ims[i] = dwi_im 
        end_time = time.time()
        logging.info("Done Gibbs unringing. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    else:
        logging.info("Skipping Gibbs unringing.")
    
    # 6. Concatenate DWIs and extract shells 
    dwis = [(d,b,v) for d,b,v,k in zip(dwi_ims, bval_arrays, bvec_arrays, keep_dwis) if k]
    if len(dwis) > 1:
        dwi_proc_im, bvals, bvecs = concatenate_dwis(*dwis)
    else:
        dwi_proc_im, bvals, bvecs = dwis[0]
    # extract shell if option provided 
    if extract_shell: 
        logging.info(f"Extract shells from multishell DWI: {extract_shell}")
        dwi_proc_im, bvals, bvecs = extract_shells_from_multishell_dwi(dwi_proc_im, bvals, bvecs, extract_shell)
    
    # 7. Eddy 
    if no_moco:
        logging.info("Skipping motion correction.")
    elif topup:
        start_time = time.time()
        if mask is None:
            mask_im = bet2_mask_nifti(unwarped_b0_im, erode_iterations=0, f=bet_f, g=bet_g)
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
            mask_im = bet2_mask_nifti(b0_im, erode_iterations=0, f=bet_f, g=bet_g)
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
    
    # 8. Resample DWI
    if resample > 0:
        resample=[resample]*3
        logging.info('Resample image to {} x {} x {}'.format(*resample))
        dwi_proc_im = resample_image(dwi_proc_im, resample, interp='BSpline')
        if bias_mask is not None:
            logging.info('Resample bias_mask to {} x {} x {}'.format(*resample))
            bias_mask_im = resample_image(bias_mask_im, resample, interp='NearestNeighbor')
        if mask:
            logging.info('Resample brain mask to {} x {} x {}'.format(*resample))
            mask_im = resample_image(mask_im, resample, interp='NearestNeighbor')
    
    # 9. Bias correction 
    if bias_corr: 
        if bias_mask is None:
            if mask is None:
                logging.info("Mask B0 with BET2 f={f} g={g} before bias field correction".format(f=bet_f, g=bet_g))
                b0_im = extract_b0(dwi_proc_im, bvals, first=True)
                mask_im = bet2_mask_nifti(b0_im, erode_iterations=0, f=bet_f, g=bet_g)
            logging.info('Bias correct within brain mask with N4BiasFieldCorrection')
            dwi_n4, bias_im = n4_bias_correct_dwi(dwi_proc_im, bvals, bvecs, field=True, mask_img=mask_im)
            dwi_proc_im, __, __ = dwi_n4 
        else:
            logging.info('Bias correct within user provided bias mask with N4BiasFieldCorrection')
            dwi_n4, bias_im = n4_bias_correct_dwi(dwi_proc_im, bvals, bvecs, field=True, mask_img=bias_mask_im)
            dwi_proc_im, __, __ = dwi_n4
        write_nifti(bias_filename, bias_im)
    
    # 10. Mask B0 and erode mask
    if mask is None:
        logging.info("Mask B0 with BET2 f={f} g={g}".format(f=bet_f, g=bet_g))
        b0_im = extract_b0(dwi_proc_im, bvals, first=no_moco, average=not no_moco)
        mask_im = bet2_mask_nifti(b0_im, erode_iterations=1, f=bet_f, g=bet_g)

    logging.info('Write mask Nifti to file {}'.format(mask_filename))
    write_nifti(mask_filename, mask_im)
        
    # 11. Save DWI and extract final B0. 
    logging.info("Save processed DWI")    
    write_dwi(dwi_processed_filename, dwi_proc_im, bvals, bvecs)
    logging.info('Extract preprocessed B0 image')
    b0_im = extract_b0(dwi_proc_im, bvals)
    write_nifti(b0_filename, b0_im)
    
    # 12. Estimate tensor
    dwi_proc_im, bvals, bvecs = extract_gaussian_shells(dwi_proc_im, bvals, bvecs)
    logging.info("Estimate tensor using WLS fit")
    tensor_im = estimate_tensor(dwi_proc_im, mask_im, bvals, bvecs, fit_method='WLS')
    TSC = TensorScalarCalculator(tensor_im, mask_im=mask_im)
    
    # 13. Save tensor files
    logging.info("Save files")
    write_nifti(tensor_filename, tensor_im)
    write_nifti(fa_filename, TSC.FA)
    write_nifti(tr_filename, TSC.TR)
    write_nifti(md_filename, TSC.MD)
    write_nifti(ax_filename, TSC.AX)
    write_nifti(rad_filename, TSC.RAD)
    
if __name__ == '__main__': 
    main(sys.argv[1:])
