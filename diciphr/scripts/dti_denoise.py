#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_dwi, write_dwi, write_nifti 
from diciphr.diffusion import concatenate_dwis, lpca_denoise, mppca_denoise, gibbs_unringing

DESCRIPTION = '''
    Denoise DWI filenames with DIPY's Local PCA denoising.
'''

PROTOCOL_NAME='Denoise_DWI'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', action='store', metavar='<dwi>', dest='dwi_filenames',
                    type=str, required=True, nargs="*",
                    help='Path(s) of the DWI image file. Separate by spaces if multiple files. Associated bval/bvec files must exist'
                    )
    p.add_argument('-o', action='store', metavar='<nii>', dest='output_filename',
                    type=str, required=True, 
                    help='Name of output DWI file. bval/bvec files will be written with corresponding basename.'
                    )
    p.add_argument('-f', action='store', metavar='<nii>', dest='diff_filename',
                    type=str, required=False, default=None,
                    help='Path to save difference map between input DWI and denoised DWI.'
                    )
    p.add_argument('-m', '--method', action='store', metavar='<str>', dest='method',
                    type=str, required=False, default='mppca',
                    help='Denoising method, options are mppca (Default) and lpca.'                    
                    )
    p.add_argument('-g', '--gibbs', action='store_true', dest='gibbs', 
                    help='Apply Gibbs de-ringing algorithm'
                    )
    p.add_argument('-s', '--slice', action='store', dest='acquisition_slicetype',
                    type=str, required=False, default='axial', 
                    help='The slice type acquired by the MRI, used for Gibbs de-ringing. Default: axial'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_filename))
    make_dir(output_dir,recursive=True,pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(*args.dwi_filenames, nifti=True)
        dwis = [ read_dwi(fn) for fn in args.dwi_filenames ]
        dwi, bvals, bvecs = concatenate_dwis(*dwis)
        
        if args.method.lower() == 'mppca':
            logging.info("Denoise DWI data with MP-PCA algorithm")
            dwi_denoised_im, dwi_denoise_diff = mppca_denoise(dwi, bvals, bvecs, return_diff=True) 
        elif args.method.lower() == 'lpca':
            logging.info("Denoise DWI data with LPCA algorithm")
            dwi_denoised_im, dwi_denoise_diff = lpca_denoise(dwi, bvals, bvecs, return_diff=True)
        else:
            raise ValueError('Denoising method not recognized: {}'.format(args.method))
        if args.gibbs:
            logging.info("Correct DWI data for Gibbs ringing artifact")
            dwi_proc_im, dwi_gibbs_diff_im = gibbs_unringing(dwi_denoised_im, 
                acquisition_slicetype=args.acquisition_slicetype, return_diff=True)
        else:
            dwi_proc_im = dwi_denoised_im 
        logging.info("Saving DWI files after denoising steps")
        write_dwi(args.output_filename, dwi_proc_im, bvals, bvecs)
        if args.diff_filename:
            write_nifti(args.diff_filename, dwi_denoise_diff) 
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

if __name__ == '__main__':
    main(sys.argv[1:])