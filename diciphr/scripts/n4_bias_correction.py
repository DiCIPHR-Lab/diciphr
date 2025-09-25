#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, which, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti, read_dwi, write_nifti, write_dwi, n4_bias_correct
from diciphr.diffusion import n4_bias_correct_dwi

DESCRIPTION = '''
    A basic wrapper for ANTs N4BiasFieldCorrection. If a DWI input is provided, 
    apply N4BiasFieldCorrection to the B0 image to generate the bias field map, 
    and then divide each volume by the field map to correct the entire series.
'''

PROTOCOL_NAME='N4_Bias_Correction'   

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i',action='store',metavar='<nii>',dest='nifti_file',
                    type=str, required=True, 
                    help='The input Nifti filename. Can be a DWI nifti file if corresponding bval/bvec files exist'
                    )
    p.add_argument('-o',action='store',metavar='<nii>',dest='output',
                    type=str, required=True, 
                    help='The output Nifti filename for the bias corrected image.'
                    )
    p.add_argument('-f',action='store',metavar='<nii>',dest='output_field',
                    type=str, required=False, default=None,
                    help='The output Nifti filename for the bias field map.'
                    )
    p.add_argument('-w',action='store',metavar='<nii>',dest='weight_nii',
                    type=str, required=False, default=None,
                    help='Perform a relative weighting of specific voxels during the B-spline fitting'
                    )
    p.add_argument('-x',action='store',metavar='<nii>',dest='mask_nii',
                    type=str, required=False, default=None,
                    help='The final bias correction is only performed in the mask region.'
                    )
    p.add_argument('-c', action='store', metavar='[<nIters>,<threshold>]', dest='convergence',
                    type=str, required=False, default=None,
                    help='Convergence argument to pass to ANTs. See usage for N4BiasFieldCorrection for details. Default is dependent on input modality (DWI or other type of image)'
                    )
    p.add_argument('-b', action='store',metavar='[<distance>,<resolution>]',dest='bspline',
                    type=str, required=False, default=None,
                    help='BSpline fitting argument to pass to ANTs. See usage for N4BiasFieldCorrection for details. Default is dependent on input modality (DWI or other type of image)'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        logging.debug("Check for executables in path")
        which('N4BiasFieldCorrection')
        check_inputs(args.nifti_file, nifti=True)
        try:
            nifti_img, bvals, bvecs = read_dwi(args.nifti_file)
            diffusion_input = True
        except DiciphrException:
            nifti_img = read_nifti(args.nifti_file)
            diffusion_input = False
        if args.mask_nii:
            check_inputs(args.mask_nii, nifti=True)
            mask_img = read_nifti(args.mask_nii)
        else:
            mask_img = None
        if args.weight_nii:
            check_inputs(args.weight_nii, nifti=True)
            weight_img = read_nifti(args.weight_nii)
        else:
            weight_img = None
        # Run the N4 function 
        field_option = (args.output_field is not None)
        if diffusion_input:
            # DWI 
            result = n4_bias_correct_dwi(nifti_img, bvals, bvecs, 
                field=field_option, mask_img=mask_img, weight_img=weight_img,
                convergence=args.convergence, bspline=args.bspline                        
                )
        else:
            # non-DWI
            result = n4_bias_correct(nifti_img, 
                field=field_option, mask_img=mask_img, weight_img=weight_img,
                convergence=args.convergence, bspline=args.bspline                        
                )
        if field_option:
            # result is a tuple of 2 
            result, field_img = result[1]
            write_nifti(args.output_field, field_img)
        if diffusion_input:
            # DWI - result is a tuple of 3
            dwi_img, bvals, bvecs = result
            write_dwi(args.output, dwi_img, bvals, bvecs)
        else:
            # non-DWI
            write_nifti(args.output, result)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

if __name__ == '__main__': 
    main(sys.argv[1:])
