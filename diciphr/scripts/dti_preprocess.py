#! /usr/bin/env python

import os, sys, logging, time
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
               dwi_filenames_from_directory, DiciphrArgumentParser )
from diciphr.nifti_utils import ( read_nifti, write_nifti, read_dwi, write_dwi,
               mask_nifti, json_files_from_niftis, 
               resample_image, split_image )
from diciphr.diffusion import ( concatenate_dwis, round_bvals, extract_b0, 
               extract_shells_from_multishell_dwi, extract_gaussian_shells, 
               mppca_denoise, gibbs_unringing, n4_bias_correct_dwi, 
               prepare_acqparams_json, prepare_acqparams_nojson, unique_acqparams, 
               phase_enc_from_json, synb0_disco, run_topup_post_synb0, 
               pad_image_for_topup, crop_image_post_topup, run_topup, 
               apply_topup, fsl_eddy, fsl_eddy_post_topup, save_eddy_text, 
               estimate_tensor, TensorScalarCalculator )
from diciphr.registration import ants_registration_dti_t1

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
    g_oio.add_argument('-t', '--t1', action='store', metavar='<t1>', dest='t1', 
                    type=str, required=False, default=None, 
                    help='Provide a T1 image to register to DTI space and/or run SynB0-Disco'
                    )
    g_oio.add_argument('--t1-mask', action='store', metavar='<mask>', dest='t1_mask', 
                    type=str, required=False, default=None, 
                    help='Provide a mask for T1 image to register to DTI space and/or run SynB0-Disco'
                    )    
    g_oio.add_argument('--synb0', action='store_true', dest='run_synb0', 
                    help='Run SynB0-Disco with the T1'
                    ) 
    g_oio.add_argument('--topup', action='store', metavar='<topup>', dest='topup',
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
    g_dn.add_argument('-N', '--no-denoise', action='store_false', dest='denoise',
                    help='Skip MPPCA denoising'
                    )
    g_dn.add_argument('-G', '--gibbs', action='store_true', dest='gibbs',
                    help='Run Gibbs unringing on the data (default: False)'
                    )
    g_dn.add_argument('-l', '--acquisition-slice', action='store', dest='acquisition_slicetype', 
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
    g_e.add_argument('--no-json', action='store_true', dest='no_json', 
                    help='Do not attempt to extract total readout time and phase encoding direction from json files'
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
    g_o.add_argument('--mask-method', action='store', dest='mask_method',
                    type=str, required=False, default='synthstrip', 
                    help='Method used for masking the B0 image. Options are synthstrip(default), bet.'
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
    try:
        run_dti_preprocess(args)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def initialize_and_validate_args(args):
    """
    Normalize arguments, resolve inputs, perform sanity checks,
    set up logging, and log configuration.
    Returns
    -------
    args : argparse.Namespace
        Normalized and validated arguments
    """
    # Resolve output directory 
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.getcwd(), args.subject, "DTI_Preprocess"
        )
    make_dir(args.output_dir, recursive=True, pass_if_exists=True)

    # Normalize basic args 
    if args.extract_shell and 0 not in args.extract_shell:
        args.extract_shell = [0] + list(args.extract_shell)
    args.mask_method = args.mask_method.lower()
    if args.mask_method == "bet":
        args.mask_kwargs = {"erode_iterations": 1, "f": 0.2, "g": 0.0}
        args.bias_mask_kwargs = {"erode_iterations": 0, "f": 0.2, "g": 0.0}
    elif args.mask_method == "synthstrip":
        args.mask_kwargs = {"border": 0, "fill": 0, "no_csf": False}
        args.bias_mask_kwargs = {"border": 2, "fill": 0, "no_csf": False}
    else:
        raise ValueError(f"Mask method not recognized: {args.mask_method}")

    # Initialize logging 
    protocol_logging(
        PROTOCOL_NAME, directory=(args.logdir or args.output_dir),
        filename=args.logfile, debug=args.debug, create_dir=True
    )

    # Resolve DWI filenames 
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

    # Optional input checks 
    if args.mask:
        check_inputs(args.mask, nifti=True)
    if args.bias_mask:
        check_inputs(args.bias_mask, nifti=True)
    if args.bias_corr:
        args.bias_iterations = list(map(int, args.bias_iterations.split(",")))

    # Topup input validation
    if args.topup:
        # user supplied topup prefix
        check_inputs(
            args.topup + "_fieldcoef.nii.gz",
            args.topup + "_movpar.txt",
        )
        if args.index:
            check_inputs(args.index)
        elif not args.no_moco:
            check_inputs(args.topup + "_index.txt")
        if args.acqparams:
            check_inputs(args.acqparams)
        else:
            check_inputs(args.topup + "_acqparams.txt")
        logging.info("Previous FSL topup outputs found")

    # ---------- Log configuration ----------
    logging.info(f"subject: {args.subject}")
    logging.info(f"dwi_filenames: {args.dwi_filenames}")
    logging.info(f"output_dir: {args.output_dir}")
    if args.resample > 0:
        logging.info(f"resample: {args.resample}")
    else:
        logging.info("Not resampling data")

    if args.mask:
        logging.info(f"Mask: {args.mask}")
    if args.t1:
        logging.info(f"T1: {args.t1}")
    if args.t1_mask:
        logging.info(f"T1 mask: {args.t1_mask}")
    if args.topup:
        logging.info(f"Topup: {args.topup}")
    if args.extract_shell:
        logging.info(f"Extract shell: {args.extract_shell}")
    if args.no_moco:
        logging.info("Skipping motion correction")
    if not args.denoise:
        logging.info("Skipping denoising")
    logging.info(f"Masking method: {args.mask_method}")
    logging.debug(f"args: {args}")
    return args

def build_output_filenames(args):
    logging.debug(f"build_output_filenames")
    outputs = {}
    prefix = os.path.join(args.output_dir, args.subject)
    outputs['dwi_processed_filename'] = f"{prefix}_DWI_preprocessed.nii.gz"
    outputs['mask_filename'] = f"{prefix}_tensor_mask.nii.gz"
    outputs['tensor_filename'] = f"{prefix}_tensor.nii.gz"
    outputs['fa_filename'] = f"{prefix}_tensor_FA.nii.gz"
    outputs['tr_filename'] = f"{prefix}_tensor_TR.nii.gz"
    outputs['md_filename'] = f"{prefix}_tensor_MD.nii.gz"
    outputs['ax_filename'] = f"{prefix}_tensor_AX.nii.gz"       
    outputs['rad_filename'] = f"{prefix}_tensor_RAD.nii.gz"    
    outputs['b0_filename'] = f"{prefix}_B0.nii.gz"
    outputs['eddy_text_prefix'] = f"{prefix}"  
    outputs['bias_filename'] = f"{prefix}_bias_field.nii.gz"
    outputs['topup_base'] = f"{prefix}_topup"
    outputs['registration_prefix'] = f"{prefix}"
    outputs['synb0_output_file'] = f"{prefix}_synb0.nii.gz"
    logging.debug(f"outputs: {outputs}")
    return outputs

def load_inputs(args):
    logging.debug("load_inputs")
    logging.info('Load dwi_filenames')
    dwi_ims = []
    bval_arrays = []
    bvec_arrays = []
    for dwifn in args.dwi_filenames:
        _d, _b, _v = read_dwi(dwifn, force=True)
        dwi_ims.append(_d)
        bval_arrays.append(round_bvals(_b))
        bvec_arrays.append(_v)
    if args.mask:
        logging.info('Load user provided mask')
        mask_im = read_nifti(args.mask)
    else:
        mask_im = None 
    if args.bias_mask:
        logging.info('Load user provided bias weight mask')
        bias_mask_im = read_nifti(args.bias_mask)
    else:
        bias_mask_im = None
    if args.t1 or args.t1_mask:
        logging.info('Load optional T1 inputs')
    if args.t1:
        t1_im = read_nifti(args.t1)
    else:
        t1_im = None 
    if args.t1_mask:
        t1_mask_im = read_nifti(args.t1_mask)
    else:
        t1_mask_im = None    
    shapes = [im.shape[:3] for im in dwi_ims]
    if len(set(shapes)) != 1:
        raise ValueError(f"All DWI inputs must have identical spatial shape; got {shapes}")    
    if mask_im is not None and mask_im.shape[:3] != shapes[0]:
        raise ValueError("Mask spatial shape does not match DWI inputs")  
    logging.debug(f"dwi_ims: {dwi_ims}")
    logging.debug(f"bval_arrays: {bval_arrays}")
    logging.debug(f"bvec_arrays: {bvec_arrays}")
    logging.debug(f"mask_im: {mask_im}")
    logging.debug(f"bias_mask_im: {bias_mask_im}")
    logging.debug(f"t1_im: {t1_im}")
    logging.debug(f"t1_mask_im: {t1_mask_im}")
    return dwi_ims, bval_arrays, bvec_arrays, mask_im, bias_mask_im, t1_im, t1_mask_im 

def phase_enc_and_filter_dwis(args, dwi_ims, bval_arrays):
    logging.debug("phase_enc_and_filter_dwis")
    phase_encs = args.phase_encs
    if args.no_json:
        json_files = None
    else:
        json_files = json_files_from_niftis(args.dwi_filenames)
    if json_files:
        logging.info("Get acquisition parameters from .json files")
        all_acqparams = [prepare_acqparams_json(fn, dwi_im) for fn, dwi_im in zip(json_files, dwi_ims)]
        logging.debug(f'all_acqparams: {all_acqparams}')
        phase_encs = [phase_enc_from_json(fn) for fn in json_files]
        logging.debug(f'phase_encs: {phase_encs}')
    else:
        logging.info("Get acquisition parameters without .json files")
        if len(phase_encs) != len(args.dwi_filenames):
            if len(phase_encs) == 1:
                phase_encs = [phase_encs[0] for fn in args.dwi_filenames]
            elif len(phase_encs) > 1:
                raise ValueError("Number of phase encoding dirs does not match or could not be broadcast to number of DWI files")
        if len(phase_encs) > 0:
            logging.info("Get acquisition parameters without .json files")
        all_acqparams = [prepare_acqparams_nojson(args.readout_time, phase_enc) for phase_enc in phase_encs]
        logging.debug(f'all_acqparams: {all_acqparams}')
    if len(phase_encs) == 0:
        if args.run_synb0 or args.t1:
            raise ValueError('To run synb0 or register T1 image, phase encoding direction must be provided by .json file or by -P argument')        
    # Array of which DWI images to keep in output 
    keep_dwis = [len(bval[bval>0])>=6 and len(bval[bval==0])>=1 for bval in bval_arrays]
    for fn, k in zip(args.dwi_filenames, keep_dwis):
        if k or unique_acqparams(all_acqparams) == 1:
            logging.info(f"DWI {fn} will be pre-processed")
        else:
            logging.info(f"DWI {fn} will be used for topup and discarded")
    if not any(keep_dwis):
        raise ValueError("None of DWI inputs have at least 6 weighted volumes and at least 1 unweighted volume. Input is invalid")
    logging.debug(f"phase_encs: {phase_encs}")
    logging.debug(f"all_acqparams: {all_acqparams}")
    logging.debug(f"keep_dwis: {keep_dwis}")
    return phase_encs, all_acqparams, keep_dwis

def denoising_stage(args, dwi_ims, bval_arrays, bvec_arrays, keep_dwis):
    """
    Apply denoising and/or Gibbs unringing to DWIs marked as kept.

    Returns
    -------
    denoised_ims : list of nibabel.Nifti1Image
        One entry per kept DWI, in original order.
    """
    logging.debug("denoising_stage")
    denoised_ims = []
    if not args.denoise:
        logging.info("Skipping data denoising.")
    if not args.gibbs:
        logging.info("Skipping Gibbs unringing.")
    start_time = time.time()
    for dwi_im, bvals, bvecs, keep in zip(dwi_ims, bval_arrays, bvec_arrays, keep_dwis):
        if not keep:
            continue
        out_im = dwi_im
        if args.denoise:
            out_im = mppca_denoise(
                out_im, bvals, bvecs,
                patch_radius=2,
                return_diff=False
            )
        if args.gibbs:
            out_im = gibbs_unringing(
                out_im,
                acquisition_slicetype=args.acquisition_slicetype,
                n_points=3,
                num_processes=1,
                return_diff=False
            )
        denoised_ims.append(out_im)
    if args.denoise or args.gibbs:
        elapsed = (time.time() - start_time) / 60.0
        logging.info("Done denoising. Elapsed time {0:0.2f} minutes".format(elapsed))
    logging.debug(f"denoised_ims: {denoised_ims}")
    return denoised_ims

def topup_stage(args, dwi_ims, bval_arrays, bvec_arrays, all_acqparams, 
                    keep_dwis, outputs, mask_im, t1_im, t1_mask_im):
    """
    Run topup before denoising. Pads DWI images to even dimensions, saves reference images for later
    cropping, and runs or loads topup as needed.
    """
    logging.debug("topup_stage")
    # Save references before padding
    reference_im = dwi_ims[0]
    
    # Pad DWIs for topup
    dwi_ims = [pad_image_for_topup(im) for im in dwi_ims]
    topup = args.topup
    acqparams = args.acqparams
    index = args.index
    unwarped_b0_im = None
    
    if mask_im is not None:
        # Pad here if user-provided, because if not, mask
        #  will be estimated on padded grid by eddy stage 
        mask_im = pad_image_for_topup(mask_im)
        
    # Case 1: user-provided topup
    if topup:
        logging.info("Using field estimate from previously run topup")

        if not acqparams:
            acqparams = topup + "_acqparams.txt"
        if not index:
            index = topup + "_index.txt"

        b0_im = extract_b0(dwi_ims[0], bval_arrays[0], first=True)
        unwarped_b0_im = apply_topup(b0_im, topup, acqparams)

    # Case 2: multiple phase-encoding directions
    elif unique_acqparams(all_acqparams) > 1:
        logging.info("Run topup")
        topup = run_topup(
            dwi_ims, bval_arrays, bvec_arrays, all_acqparams, 
            outputs["topup_base"], keep_dwis
        )
        acqparams = topup + "_acqparams.txt"
        index = topup + "_index.txt"
        unwarped_b0_im = read_nifti(topup + "_b0u.nii.gz")
        if len(unwarped_b0_im.shape) == 4:
            unwarped_b0_im = split_image(unwarped_b0_im)[0]

    # Case 3: single phase-encoding direction
    elif args.run_synb0:
        logging.info("Run SynB0-Disco")
        synb0_img = synb0_disco(
            dwi_ims[0], bval_arrays[0], all_acqparams[0], t1_im, t1_mask=t1_mask_im
        )
        write_nifti(outputs["synb0_output_file"], synb0_img)
        # Concatenate DWIs here so that index is created with the correct number 
        dwi_concat, bvals_concat, bvecs_concat = concatenate_dwis(
                *zip(dwi_ims, bval_arrays, bvec_arrays), keep_dwis=keep_dwis
        )
        run_topup_post_synb0(
            dwi_concat, bvals_concat, bvecs_concat, synb0_img, 
            all_acqparams[0], outputs["topup_base"],
        )
        topup = outputs["topup_base"]
        acqparams = topup + "_acqparams.txt"
        index = topup + "_index.txt"
        b0_im = extract_b0(dwi_ims[0], bval_arrays[0], first=True)
        unwarped_b0_im = apply_topup(b0_im, topup, acqparams)
    logging.debug(f"dwi_ims: {dwi_ims}")
    logging.debug(f"reference_im: {reference_im}")
    logging.debug(f"topup: {topup}")
    logging.debug(f"acqparams: {acqparams}")
    logging.debug(f"index: {index}")
    logging.debug(f"unwarped_b0_im: {unwarped_b0_im}")
    return dwi_ims, reference_im, topup, acqparams, index, unwarped_b0_im, mask_im

def fill_denoised_dwis(padded_dwi_ims, denoised_ims, keep_dwis, reference_im):
    """
    Insert denoised data into the non-padded region of padded DWI images.

    Parameters
    ----------
    padded_dwi_ims : list of nibabel.Nifti1Image
        DWIs after pad_image_for_topup
    denoised_ims : list of nibabel.Nifti1Image
        Denoised DWIs in original (unpadded) space, one per kept DWI
    keep_dwis : list of bool
        Flags indicating which DWIs were denoised
    reference_im : nibabel.Nifti1Image
        Original unpadded reference image used to define non-padded region

    Returns
    -------
    out_dwi_ims : list of nibabel.Nifti1Image
        Padded DWIs with denoised data written into non-padded region
    """
    logging.debug("fill_denoised_dwis")
    out_dwi_ims = []
    j = 0
    for padded_im, keep in zip(padded_dwi_ims, keep_dwis):
        if keep:
            # Write denoised data into non-padded region
            out_im = pad_image_for_topup(
                denoised_ims[j],
                reference_img=padded_im,
            )
            j += 1
        else:
            out_im = padded_im
        out_dwi_ims.append(out_im)
    logging.debug(f"out_dwi_ims: {out_dwi_ims}")
    return out_dwi_ims

def concatenate_and_extract_shells(args, dwi_ims, bval_arrays, bvec_arrays, keep_dwis):
    """
    Concatenate kept DWIs and optionally extract shells.
    """
    logging.debug("concatenate_and_extract_shells")
    dwi_tuples = [(d, b, v) for d, b, v in zip(dwi_ims, bval_arrays, bvec_arrays)]
    dwi_proc_im, bvals, bvecs = concatenate_dwis(*dwi_tuples, keep_dwis=keep_dwis)
    if args.extract_shell:
        logging.info(f"Extract shells from multishell DWI: {args.extract_shell}")
        dwi_proc_im, bvals, bvecs = extract_shells_from_multishell_dwi(
            dwi_proc_im, bvals, bvecs, args.extract_shell
        )
    logging.debug(f"dwi_proc_im: {dwi_proc_im}")
    logging.debug(f"bvals: {bvals}")
    logging.debug(f"bvecs: {bvecs}")
    return dwi_proc_im, bvals, bvecs

def eddy_stage(args, dwi_proc_im, bvals, bvecs, mask_im, reference_im, outputs,  
                   topup, acqparams, index, unwarped_b0_im):
    """
    Run eddy motion/distortion correction stage.
    """
    logging.debug("eddy_stage")
    if args.no_moco:
        logging.info("Skipping motion correction") 
        dwi_proc_im = crop_image_post_topup(dwi_proc_im, reference_im)
        return dwi_proc_im, bvals, bvecs, mask_im
    start_time = time.time()
    # Case 1: eddy with topup
    if topup:
        if mask_im is None:
            mask_im = mask_nifti(unwarped_b0_im, method=args.mask_method,
                **args.mask_kwargs)
        logging.info("Run eddy with topup")
        dwi_proc_im, bvals, bvecs, eddy_text_outputs = fsl_eddy_post_topup(
            dwi_proc_im, bvals, bvecs, topup, acqparams, index, mask_im,
            unwarped_b0_im=unwarped_b0_im, replace_outliers=args.replace_outliers)
    # Case 2: eddy without topup
    else:
        if mask_im is None:
            b0_im = extract_b0(dwi_proc_im, bvals, first=True)
            mask_im = mask_nifti(b0_im, method=args.mask_method,
                **args.bias_mask_kwargs)
        logging.info("Run eddy without topup")
        dwi_proc_im, bvals, bvecs, eddy_text_outputs = fsl_eddy(
            dwi_proc_im, bvals, bvecs, mask_im, readout_time=args.readout_time,
            replace_outliers=args.replace_outliers)
    save_eddy_text(outputs["eddy_text_prefix"], eddy_text_outputs)
    elapsed = (time.time() - start_time) / 60.0
    logging.info(
        "Done eddy. Elapsed time {0:0.2f} minutes".format(elapsed)
    )
    # Crop back to original (pre-padding) shape exactly once
    dwi_proc_im = crop_image_post_topup(dwi_proc_im, reference_im)
    mask_im = crop_image_post_topup(mask_im, reference_im)
    logging.debug(f"dwi_proc_im: f{dwi_proc_im}")
    logging.debug(f"bvals: {bvals}")
    logging.debug(f"bvecs: {bvecs}")
    logging.debug(f"mask_im: {mask_im}")
    return dwi_proc_im, bvals, bvecs, mask_im

def resampling_stage(args, dwi_proc_im, mask_im, bias_mask_im):    
    """
    Resample DWI and associated masks to isotropic resolution.
    """  
    logging.debug("resampling_stage")
    if args.resample <= 0:
        return dwi_proc_im, mask_im, bias_mask_im
    spacing = [args.resample] * 3
    logging.info("Resample image to {} x {} x {}".format(*spacing))
    dwi_proc_im = resample_image(dwi_proc_im, spacing, interp="Linear")
    if bias_mask_im is not None:
        logging.info("Resample bias_mask to {} x {} x {}".format(*spacing))
        bias_mask_im = resample_image(bias_mask_im, spacing, 
                                      interp="NearestNeighbor")
    if mask_im is not None:
        logging.info("Resample brain mask to {} x {} x {}".format(*spacing))
        mask_im = resample_image(mask_im, spacing, interp="NearestNeighbor")
    logging.debug(f"dwi_proc_im: {dwi_proc_im}")
    logging.debug(f"mask_im: {mask_im}")
    logging.debug(f"bias_mask_im: {bias_mask_im}")
    return dwi_proc_im, mask_im, bias_mask_im

def bias_correction_stage(args, dwi_proc_im, bvals, bvecs, mask_im, 
                              bias_mask_im, outputs):
    """
    Run N4 bias field correction on the DWI.
    """
    logging.debug("bias_correction_stage")
    if not args.bias_corr:
        return dwi_proc_im, None, mask_im
    # Case 1: no user-provided bias mask
    if bias_mask_im is None:
        if mask_im is None:
            logging.info(f"Mask B0 with {args.mask_method} before bias field correction")
            b0_im = extract_b0(dwi_proc_im, bvals, first=True)
            mask_im = mask_nifti(b0_im, method=args.mask_method, 
                                 **args.bias_mask_kwargs)
        logging.info("Bias correct within brain mask with N4BiasFieldCorrection")
        (dwi_corr, _, _), bias_im = n4_bias_correct_dwi(
            dwi_proc_im, bvals, bvecs, field=True, mask_img=mask_im
        )
        dwi_proc_im = dwi_corr
    # Case 2: user-provided bias mask
    else:
        logging.info("Bias correct within user provided bias mask with N4BiasFieldCorrection")
        (dwi_corr, _, _), bias_im = n4_bias_correct_dwi(
            dwi_proc_im, bvals, bvecs, field=True, mask_img=bias_mask_im,
        )
        dwi_proc_im = dwi_corr
    logging.info("Write bias image to file")
    write_nifti(outputs['bias_filename'], bias_im)
    logging.debug(f"dwi_proc_im: {dwi_proc_im}")
    logging.debug(f"bias_im: {bias_im}")
    logging.debug(f"mask_im: {mask_im}")
    return dwi_proc_im, bias_im, mask_im

def mask_b0_stage(args, dwi_proc_im, bvals, mask_im, outputs):
    """
    Extract B0 image and create brain mask if one was not provided.
    """
    logging.debug("mask_b0_stage")
    # Extract B0
    b0_im = extract_b0(dwi_proc_im, bvals, first=args.no_moco,
        average=(not args.no_moco))
    # Create mask only if user did not provide one
    if mask_im is None:
        logging.info(f"Mask B0 with {args.mask_method}")
        mask_im = mask_nifti(b0_im, method=args.mask_method, **args.mask_kwargs)
    logging.info(f"Write mask image to file")
    write_nifti(outputs["mask_filename"], mask_im)
    logging.info(f"Write B0 image to file")
    write_nifti(outputs['b0_filename'], b0_im)
    logging.debug(f"b0_im: {b0_im}")
    logging.debug(f"mask_im: {mask_im}")
    return b0_im, mask_im

def tensor_fitting_stage(args, dwi_proc_im, bvals, bvecs, mask_im, outputs):
    """
    Estimate diffusion tensor and write tensor-derived scalar maps, and save.
    """
    logging.debug("tensor_fitting_stage")
    # Restrict to Gaussian shells only
    dwi_proc_im, bvals, bvecs = extract_gaussian_shells(
        dwi_proc_im, bvals, bvecs
    )
    logging.info("Estimate tensor using WLS fit")
    tensor_im = estimate_tensor(
        dwi_proc_im, mask_im, bvals, bvecs, fit_method="WLS"
    )

    TSC = TensorScalarCalculator(
        tensor_im, mask_im=mask_im,
    )

    logging.info("Save tensor and scalar maps")
    write_nifti(outputs["tensor_filename"], tensor_im)
    write_nifti(outputs["fa_filename"], TSC.FA)
    write_nifti(outputs["tr_filename"], TSC.TR)
    write_nifti(outputs["md_filename"], TSC.MD)
    write_nifti(outputs["ax_filename"], TSC.AX)
    write_nifti(outputs["rad_filename"], TSC.RAD)
    fa_im = TSC.FA
    logging.debug(f"tensor_im: {tensor_im}")
    logging.debug(f"fa_im: {fa_im}")
    return tensor_im, fa_im

def registration_stage(args, b0_im, t1_im, t1_mask_im, fa_img, mask_im, outputs,
    topup, phase_encs):
    """
    Register DTI to T1 space.
    """
    logging.debug("registration_stage")
    if not args.t1:
        return
    if t1_mask_im is None:
        logging.info("Skull strip the T1 before registration")
        t1_im, t1_mask_im = mask_nifti(
            t1_im, method=args.mask_method, return_brain=True,
        )
    logging.info("Register DTI to T1")
    ants_registration_dti_t1(
        outputs["registration_prefix"], b0_im, t1_im, fa_img=fa_img,
        dti_mask_img=mask_im, syn=(topup is None), phase_enc=phase_encs[0]
    )
    
def run_dti_preprocess(args):
    """
    Run the DiCIPHR DTI Processing pipeline.    
    """    
    # 0. First steps 
    args = initialize_and_validate_args(args)
    outputs = build_output_filenames(args)
    
    # 1. Load inputs
    ( dwi_ims, bval_arrays, bvec_arrays, mask_im, bias_mask_im, 
        t1_im, t1_mask_im ) = load_inputs(args)
    
    # 2. Get acquisition parameters from json files or from command line 
    phase_encs, all_acqparams, keep_dwis = phase_enc_and_filter_dwis(
        args, dwi_ims, bval_arrays
    )
    
    # 3. Denoise the images which will be kept 
    denoised_ims = denoising_stage(
        args, dwi_ims, bval_arrays, bvec_arrays, keep_dwis
    )
    
    # 4. Topup on the pre-denoised data - dwi_ims may now be padded  
    ( dwi_ims, reference_im, topup, acqparams, index, unwarped_b0_im, mask_im 
     ) = topup_stage(
        args, dwi_ims, bval_arrays, bvec_arrays, all_acqparams, 
        keep_dwis, outputs, mask_im, t1_im, t1_mask_im
    )

    # 5. Fill in the data from denoised_ims into their proper place in possibly padded array 
    dwi_ims = fill_denoised_dwis(
        dwi_ims, denoised_ims, keep_dwis, reference_im
    )
    
    # 6. Concatenate DWIs and extract shells 
    dwi_proc_im, bvals, bvecs = concatenate_and_extract_shells(
        args, dwi_ims, bval_arrays, bvec_arrays, keep_dwis
    )
    
    # 7. Eddy and final cropping if needed 
    dwi_proc_im, bvals, bvecs, mask_im = eddy_stage(
        args, dwi_proc_im, bvals, bvecs, mask_im, reference_im, outputs, 
        topup, acqparams, index, unwarped_b0_im
    )

    # 8. Optionally resample DWI
    dwi_proc_im, mask_im, bias_mask_im = resampling_stage(
        args, dwi_proc_im, mask_im, bias_mask_im
    )
    
    # 9. Bias correction
    dwi_proc_im, bias_im, mask_im = bias_correction_stage(
        args, dwi_proc_im, bvals, bvecs, mask_im, bias_mask_im, outputs, 
    )
    
    # 10. Save the DWI 
    logging.info("Save processed DWI")    
    write_dwi(outputs['dwi_processed_filename'], dwi_proc_im, bvals, bvecs)
    
    # 11. Mask B0 and erode mask
    b0_im, mask_im = mask_b0_stage(
        args, dwi_proc_im, bvals, mask_im, outputs
    )
    
    # 12. Estimate tensor
    tensor_im, fa_im = tensor_fitting_stage(
        args, dwi_proc_im, bvals, bvecs, mask_im, outputs
    )
    
    # 13. Registration 
    registration_stage(
        args, b0_im, t1_im, t1_mask_im, fa_im, mask_im, outputs, topup, phase_encs
    )
            
if __name__ == '__main__': 
    main(sys.argv[1:])