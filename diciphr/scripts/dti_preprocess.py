#! /usr/bin/env python

import os, sys, logging, traceback, argparse, time
from ..utils import ( which, check_inputs, 
                make_dir, protocol_logging )
from ..nifti_utils import ( read_nifti, write_nifti, nifti_image,
                read_dwi, write_dwi, threshold_image, resample_image, is_nifti_file )
from ..diffusion import ( concatenate_dwis, round_bvals, extract_b0, 
                extract_shells_from_multishell_dwi, extract_gaussian_shells, 
                normalize_dwi, mask_dwi, n4_bias_correct_dwi, lpca_denoise, 
                bet2_mask_nifti, fsl_eddy, fsl_eddy_post_topup, run_topup,
                estimate_tensor, TensorScalarCalculator)
from numpy import savetxt

DESCRIPTION = '''
    Performs preprocessing on one or more DWI images.
    Multiple DWI images will be concatenated.
'''

PROTOCOL_NAME='DTI_Preprocess'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    
    g_io = p.add_argument_group('input / output arguments')
    g_io.add_argument('-d', action='store', metavar='<dwi>', dest='dwi_filenames',
                    type=str, required=True, nargs="*",
                    help='Path(s) of the DWI image file. Separate by spaces if multiple files. Associated bval/bvec files must exist'
                    )
    g_io.add_argument('-o', action='store', metavar='<dir>', dest='output_dir',
                    type=str, required=False, default=None,
                    help='Name of output directory. Defaults to {}'.format(os.path.join('$PWD','Protocols',PROTOCOL_NAME))
                    )
    g_io.add_argument('-s', '--subject', action='store', metavar='<str>', dest='subject',
                    type=str, required=True,
                    help='The output basename / subject ID. Output files will be prepended with this string'
                    )
                    
    g_m = p.add_argument_group('masking options')
    g_m.add_argument('-m', '--mask', action='store', metavar='mask_filename', dest='mask_filename', 
                    type=str, required=False, default=None, 
                    help='Provide a brain mask nifti file instead of running BET'
                    )
    g_m.add_argument('--bet-f', action='store', dest='bet_f', metavar='<float>', 
                    type=float, required=False, default=0.2, 
                    help='The f parameter for BET skull stripping'
                    )
    g_m.add_argument('--bet-g', action='store', dest='bet_g', metavar='<float>', 
                    type=float, required=False, default=0.0, 
                    help='The g parameter for BET skull stripping'
                    )                
    
    g_dn = p.add_argument_group('denoising options')
    g_dn.add_argument('--no-denoise', action='store_true', dest='no_denoise',
                    required=False, default=False, 
                    help='Skip LPCA denoising'
                    )
    g_dn.add_argument('--denoise-diff', action='store_true', dest='save_denoise_diff',
                    required=False, default=False, 
                    help='Save the difference (pre-post) map of the denoising step'
                    )
                    
    g_e = p.add_argument_group('eddy options')
    g_e.add_argument('--no-moco', action='store_true', dest='no_moco',
                    required=False, default=False, 
                    help='Skip FSL eddy'
                    ) 
    g_e.add_argument('--replace-outliers', action='store_true', dest='replace_outliers',
                    required=False, default=False,
                    help='Use eddy method to replace outlier slices in the data.'
                    )    
    g_e.add_argument('-T', '--readout-time', action='store', metavar='<float>', dest='readout_time',
                    type=float, required=False, default=0.062,
                    help='Readout time (see FSL eddy documentation).'
                    )

    g_dc = p.add_argument_group('distortion correction options')              
    g_dc.add_argument('-t', '--topup', action='store', metavar='<topup>', dest='topup',
                    type=str, required=False,
                    help='Either a reverse-phase-encoding-direction DTI image OR the common prefix of results after running topup.'
                    )
    g_dc.add_argument('-p', '--phaseenc', action='store', metavar='<str>', dest='phaseenc',
                    type=str, required=False, default='AP', 
                    help='If running topup from a nifti file, provide the phase encoding direction of the DWI image.'
                    )                
    g_dc.add_argument('--index', action='store', metavar='<str>', dest='topup_index',
                    type=str, required=False, default=None,
                    help='The index.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )                
    g_dc.add_argument('--acqparams', action='store', metavar='<str>', dest='topup_acqparams',
                    type=str, required=False, default=None,
                    help='The acqparams.txt file from a previous run of topup, if not provided will ascertain from the topup prefix'
                    )                
    
    g_b = p.add_argument_group('bias correction options')
    g_b.add_argument('-b', '--bias', action='store_true', dest='bias_corr',  
                    required=False, default=False,
                    help='Bias correct the DWI with ants N4BiasFieldCorrection'
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
                    help='The threshold to pass to N4BiasFieldCorrection. Default is 0.001'
                    )  
    
    g_o = p.add_argument_group('other processing options')
    g_o.add_argument('-r', '--resample', action='store', metavar='<float>', dest='resample',
                    type=float, required=False, default=0, 
                    help='Resample DWI to an isotropic resolution. Default is 0, for no resampling.'
                    )
    g_o.add_argument('-x','--extract-shell', action='store', dest='extract_shell',
                    type=int, required=False, default=None, nargs="*",
                    help='Extract shell at this/these bvalue(s) before preprocessing'
                    )
    g_o.add_argument('--normalize', action='store_true', dest='normalize',
                    required=False, default=False,
                    help='Normalize the DWI to reference B0 value of 1000.0 before processing.'
                    )
    g_o.add_argument('--workdir', action='store', dest='workdir',
                    type=str, required=False, default=None,
                    help='The directory to preserve intermediate results. If not provided, a tempdir will be created.'
                    )
    
    g_l = p.add_argument_group('logging options')
    g_l.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    g_l.add_argument('--logfile', action='store', metavar='log', dest='logfile', 
                    type=str, required=False, default=None, 
                    help='A log file. If not provided will print to stderr.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(),'Protocols',PROTOCOL_NAME,args.subject)
    make_dir(args.output_dir,recursive=True,pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        which('bet2')
        if not args.no_moco:
            try:
                eddy_exe = which('eddy_openmp')
            except:
                eddy_exe = which('eddy') #FSL eddy must be in path.
        check_inputs(*args.dwi_filenames, nifti=True)
        if args.mask_filename:
            check_inputs(args.mask_filename, nifti=True)
        if args.bias_corr:
            which('N4BiasFieldCorrection') # N4BiasFieldCorrection must be in path.
            args.bias_iterations = list(map(int, args.bias_iterations.split(',')))
        if args.bias_mask:
            check_inputs(args.bias_mask, nifti=True)
        runtopup=False
        if args.topup:
            which('topup') 
            try:
                check_inputs(args.topup, nifti=True)
                logging.info('Nifti reverse-phase-encoding-direction file found')
                runtopup=True
            except:
                if args.topup_index is None:
                    args.topup_index = args.topup+'_index.txt'
                if args.topup_acqparams is None:
                    args.topup_acqparams = args.topup+'_acqparams.txt'
                check_inputs(args.topup+'_fieldcoef.nii.gz', args.topup+'_movpar.txt', args.topup_acqparams, args.topup_index)
                logging.info('Outputs of a previous run of FSL topup found')
        if args.workdir:
            if not os.path.exists(args.workdir):
                os.makedirs(args.workdir)
        run_dti_preprocess(args.subject, args.output_dir, args.dwi_filenames, mask=args.mask_filename, 
                        topup=args.topup, runtopup=runtopup, extract_shell=args.extract_shell,
                        normalize=args.normalize, no_moco=args.no_moco, 
                        no_denoise=args.no_denoise, save_denoise_diff=args.save_denoise_diff, 
                        bias_corr=args.bias_corr, bias_mask=args.bias_mask,
                        bias_iterations=args.bias_iterations, bias_threshold=args.bias_threshold, 
                        resample=args.resample, bet_f=args.bet_f, bet_g=args.bet_g, 
                        phaseenc=args.phaseenc, readout_time=args.readout_time, 
                        replace_outliers=args.replace_outliers, workdir=args.workdir)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
  
def run_dti_preprocess(subject, output_dir, dwi_filenames, mask=None, 
                topup=None, runtopup=False, acqparams=None, index=None, extract_shell=None, 
                normalize=False, no_moco=False, no_denoise=True, save_denoise_diff=False,
                bias_corr=False, bias_mask=None, bias_iterations=[50,50,50,50], bias_threshold=0.001, 
                resample=0, bet_f=0.2, bet_g=0.0, phaseenc="AP", readout_time=0.062, replace_outliers=False, workdir=None):
    ''' 
    Run the DTI Preprocessing protocol.
    
    Parameters
    ----------
    subject : str
        Subject ID string, used to name output files.
    output_dir : str
        Output directory.
    dwi_filenames : list
        List of filenames to subject DWI file(s).
        .bval and .bvec files must be located at same path with same basename.
    resample: Optional[float]
        Voxel dimension to resample DWI
    mask: Optional[str]
        Mask Nifti file, to use instead of running BET
    no_moco : Optional[bool]
        If True, does not run fsl eddy on the DWI image.
    no_denoise : Optional[bool]
        If True, does not run LPCA denoising on the DWI image.
    bias_corr : Optional[bool]
        If True, run ants N4BiasFieldCorrection on the DWI image. 
    bias_mask : Optional[str]
        If provided, use this mask for bias correction. 
    bet_f: Optional[float] 
        The f parameter for FSL BET2
    bet_g: Optional[float] 
        The g parameter for FSL BET2
    replace_outliers: Optional[bool]
        Replace outliers with eddy
    
    Returns
    -------
    None
    '''
    logging.info('subject: {}'.format(subject))
    logging.info('dwi_filenames: {}'.format(dwi_filenames))
    logging.info('output_dir: {}'.format(output_dir))
    if resample > 0: 
        logging.info('resample: {}'.format(resample))
    else:
        logging.info('Not resampling data.')
    if mask:
        logging.info('mask: {}'.format(mask))
    if extract_shell:
        logging.info('extract_shell: {}'.format(extract_shell))    
    if no_moco: 
        logging.info('Skipping motion correction step.')
    if no_denoise:
        logging.info('Skipping denoising step.')
    
    # Output filenames
    dwi_processed_filename = os.path.join(output_dir, '{}_DWI_preprocessed.nii.gz').format(subject)
    noise_filename = os.path.join(output_dir, '{}_denoising_diff.nii.gz').format(subject)
    mask_filename = os.path.join(output_dir, '{}_tensor_mask.nii.gz').format(subject)
    tensor_filename = os.path.join(output_dir, '{}_tensor.nii.gz').format(subject)
    fa_filename = os.path.join(output_dir, '{}_tensor_FA.nii.gz').format(subject)
    tr_filename = os.path.join(output_dir, '{}_tensor_TR.nii.gz').format(subject)        
    ax_filename = os.path.join(output_dir, '{}_tensor_AX.nii.gz').format(subject)        
    rad_filename = os.path.join(output_dir, '{}_tensor_RAD.nii.gz').format(subject)        
    b0_filename = os.path.join(output_dir, '{}_B0.nii.gz').format(subject)    
    eddy_params_filename = os.path.join(output_dir, '{}_eddy_params.txt').format(subject)    
    eddy_movement_rms_filename = os.path.join(output_dir, '{}_eddy_movement_rms.txt').format(subject)    
    eddy_restricted_movement_rms_filename = os.path.join(output_dir, '{}_eddy_restricted_movement_rms.txt').format(subject)    
    bias_filename = os.path.join(output_dir, '{}_bias_field.nii.gz').format(subject)
    
    logging.info('Begin Protocol')
    fit_method = 'WLS'
    # 1. Load dwi_filenames
    logging.info('Load dwi_filenames')
    dwi_ims = [ read_dwi(dwi) for dwi in dwi_filenames ]
    if mask:
        logging.info('Load user provided mask')
        mask_im = read_nifti(mask)
    if bias_mask:
        logging.info('Load user provided bias weight mask')
        bias_mask_im = read_nifti(bias_mask)
    
    # 2. Round bvals of all inputs 
    logging.info('Round Bvalues')
    dwi_ims = [ (dwi_im, round_bvals(bval), bvec) for (dwi_im, bval, bvec) in dwi_ims ]
    
    # 3. Concatenate DWI
    logging.info('Concatenate dwi_filenames')
    if normalize:
        logging.info('Normalize the DWIs before processing.')
        dwi_ims = [ normalize_dwi(*dwi) for dwi in dwi_ims ] 
    dwi_proc_im, bvals, bvecs = concatenate_dwis(*dwi_ims)
    logging.debug('Bvals: {}'.format(bvals))
    
    # 4. Extract single shell before anything
    if extract_shell: 
        logging.info("Extract single shell before tensor estimation")
        dwi_proc_im, bvals, bvecs = extract_shells_from_multishell_dwi(dwi_proc_im, bvals, bvecs, [0]+extract_shell)
    
    # run topup before denoising
    if topup and is_nifti_file(topup):
        revphaseenc = phaseenc[::-1]
        topup_dwi, topup_bval, topup_bvec = read_dwi(topup)
        
        # NEED TO CHECK IF DIMENSIONS ARE EVEN FOR TOPUP TO RUN
        topupshape = topup_dwi.shape
        if topupshape[0]%2 != 0 or topupshape[1]%2 != 0 or topupshape[2]%2 != 0:
            raise ValueError("Inputs to topup need to have even voxel dimensions. Crop or pad your inputs appropriately.")
        dwishape = dwi_proc_im.shape
        if dwishape[0]%2 != 0 or dwishape[1]%2 != 0 or dwishape[2]%2 != 0:
            raise ValueError("Inputs to topup need to have even voxel dimensions. Crop or pad your inputs appropriately.")
                
        topup_base = os.path.join(output_dir, subject)
        run_topup([dwi_proc_im, topup_dwi], [bvals, topup_bval], [bvecs, topup_bvec], 
            topup_base, phase_encs=[phaseenc, revphaseenc], readout_time=readout_time)
        topup = topup_base 
    
    # 5. Denoising
    if no_denoise:
        logging.info("Skipping data denoising.")
    else:
        logging.info("Denoising DWI Volume with LPCA")
        start_time = time.time()
        # DWI with more than 125 volumes produce a ValueError, that this 
        # would result in an ill-conditioned PCA matrix, increase patch_radius 
        if dwi_proc_im.shape[-1] > 125:
            patch_radius = 3
        else:
            patch_radius = 2 
        dwi_proc_im, dwi_denoise_diff_im = lpca_denoise(dwi_proc_im, bvals, bvecs, patch_radius=patch_radius, return_diff=True)
        if save_denoise_diff:
            dwi_denoise_diff_im.to_filename(noise_filename)  
        end_time = time.time()
        logging.info("Done Denoising. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    
    # 6. Eddy 
    if no_moco:
        logging.info("Skipping motion correction.")
    elif topup:
        start_time = time.time()
        if not index:
            index = topup+'_index.txt'
        if not acqparams:
            acqparams = topup+'_acqparams.txt'
            
        b0_im = extract_b0(dwi_proc_im, bvals, first=True)
        bet_mask_im = bet2_mask_nifti(b0_im, erode_iterations=1, f=bet_f, g=bet_g)
        dwi_proc_im, bvals, bvecs, eddy_params, eddy_movement_rms, eddy_restricted_movement_rms = fsl_eddy_post_topup(
                    dwi_proc_im, bvals, bvecs, topup, acqparams, index, bet_mask_im, 
                    replace_outliers=replace_outliers, workdir=workdir)
        savetxt(eddy_params_filename, eddy_params, delimiter=' ', fmt='%0.12f')
        savetxt(eddy_movement_rms_filename, eddy_movement_rms, delimiter=' ', fmt='%0.12f')
        savetxt(eddy_restricted_movement_rms_filename, eddy_restricted_movement_rms, delimiter=' ', fmt='%0.12f')
        if workdir is not None: 
            write_dwi(os.path.join(workdir, '{}_DWI_eddy.nii.gz').format(subject), dwi_proc_im, bvals, bvecs)
        end_time = time.time()
        logging.info("Done eddy. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    else:   
        logging.debug("Mask B0 with BET2 f={f} g={g} for eddy".format(f=bet_f, g=bet_g))
        start_time = time.time()
        b0_im = extract_b0(dwi_proc_im, bvals, first=True)
        bet_mask_im = bet2_mask_nifti(b0_im, erode_iterations=1, f=bet_f, g=bet_g)
        logging.info("Run FSL eddy to correct for eddy and subject motion")
        dwi_proc_im, bvals, bvecs, eddy_params, eddy_movement_rms, eddy_restricted_movement_rms = fsl_eddy(
                    dwi_proc_im, bvals, bvecs, 
                    bet_mask_im, 
                    readout_time=readout_time, 
                    replace_outliers=replace_outliers, 
                    workdir=workdir
        )
        savetxt(eddy_params_filename, eddy_params, delimiter=' ', fmt='%0.12f')
        savetxt(eddy_movement_rms_filename, eddy_movement_rms, delimiter=' ', fmt='%0.12f')
        savetxt(eddy_restricted_movement_rms_filename, eddy_restricted_movement_rms, delimiter=' ', fmt='%0.12f')
        if workdir is not None: 
            write_dwi(os.path.join(workdir, '{}_DWI_eddy.nii.gz').format(subject), dwi_proc_im, bvals, bvecs)
        end_time = time.time()
        logging.info("Done eddy. Elapsed time {0:0.2f} minutes".format((end_time - start_time)/60.))
    
    # 7. Resample DWI
    if resample > 0:
        resample=[resample]*3
        logging.info('Resample image to {} x {} x {}'.format(*resample))
        dwi_proc_im = resample_image(dwi_proc_im, resample, interp='BSpline')
        if bias_mask is not None:
            logging.info('Resample bias_mask to {} x {} x {}'.format(*resample))
            bias_mask_im = resample_image(bias_mask_im, resample, interp='NearestNeighbor')
        if mask is not None:
            logging.info('Resample brain mask to {} x {} x {}'.format(*resample))
            mask_im = resample_image(mask_im, resample, interp='NearestNeighbor')
    image_dims = dwi_proc_im.header.get_zooms()[:3]
   
    # 8. Mask B0 and erode mask
    if not mask:
        logging.info("Mask B0 with BET2 f={f} g={g}".format(f=bet_f, g=bet_g))
        b0_im = extract_b0(dwi_proc_im, bvals, first=True)
        mask_im = bet2_mask_nifti(b0_im, erode_iterations=1, f=bet_f, g=bet_g)
    logging.info('Write mask Nifti to file {}'.format(mask_filename))
    write_nifti(mask_filename, mask_im)
    
    # 9. Bias correction 
    if bias_corr: 
        if bias_mask is None:
            logging.info('Bias correct with brain mask')
            dwi_proc_im, bias_im = n4_bias_correct_dwi(dwi_proc_im, bvals, bvecs, mask_im, iterations=bias_iterations, threshold=bias_threshold, return_field=True)
        else:
            logging.info('Bias correct with user provided bias mask')
            dwi_proc_im, bias_im = n4_bias_correct_dwi(dwi_proc_im, bvals, bvecs, bias_mask_im, iterations=bias_iterations, threshold=bias_threshold, return_field=True)
        write_nifti(bias_filename, bias_im)

    # 10. Save DWI and extract final B0. 
    logging.info("Save processed DWI")    
    write_dwi(dwi_processed_filename, dwi_proc_im, bvals, bvecs)
    logging.info('Extract preprocessed B0 image')
    b0_im = extract_b0(dwi_proc_im, bvals)
    write_nifti(b0_filename, b0_im)
    
    # 11. Estimate tensor.
    dwi_proc_im, bvals, bvecs = extract_gaussian_shells(dwi_proc_im, bvals, bvecs)
    logging.info("Estimate tensor using {} fit".format(fit_method))
    tensor_im, __, __ = estimate_tensor(dwi_proc_im, mask_im, bvals, bvecs, fit_method=fit_method)
    TSC = TensorScalarCalculator(tensor_im, mask_im=mask_im)
    
    # 12. Save tensor files
    logging.info("Save files")
    write_nifti(tensor_filename, tensor_im)
    write_nifti(fa_filename, TSC.FA)
    write_nifti(tr_filename, TSC.TR)
    write_nifti(ax_filename, TSC.AX)
    write_nifti(rad_filename, TSC.RAD)
    
if __name__ == '__main__': 
    main(sys.argv[1:])
