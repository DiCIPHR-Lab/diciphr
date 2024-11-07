#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from ..utils import ( check_inputs, make_dir, make_temp_dir, which,
                protocol_logging, DiciphrException, ExecCommand, is_flirt_mat_file )
from ..nifti_utils import read_nifti, read_dwi, nifti_image, write_nifti, write_dwi
from ..diffusion import extract_b0
from numpy import percentile

DESCRIPTION = '''
    A basic wrapper for ANTs N4BiasFieldCorrection. If a DWI input is provided, 
    apply N4BiasFieldCorrection to the B0 image to generate the bias field map, 
    and then divide each volume by the field map to correct the entire series.
'''

PROTOCOL_NAME='N4_Bias_Correction'   

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
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
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        logging.debug("Check for executables in path")
        which('N4BiasFieldCorrection')
        check_inputs(args.nifti_file, nifti=True)
        if args.mask_nii:
            check_inputs(args.mask_nii, nifti=True)
        if args.weight_nii:
            check_inputs(args.weight_nii, nifti=True)
        check_inputs(output_dir, directory=True)
        run_ants_N4(args.nifti_file, args.output, field=args.output_field, 
                    mask_nii=args.mask_nii, weight_nii=args.weight_nii,
                    convergence=args.convergence, bspline=args.bspline                        
        )
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
def run_ants_N4(nifti_file, output, field=None, mask_nii=None, weight_nii=None,
                convergence=None, bspline=None):
    ''' 
    Apply N4 correction on a nifti file.
    '''
    logging.debug('Making a temporary directory')
    tmpdir = make_temp_dir(prefix=PROTOCOL_NAME)
    
    try:
        dwi_im, bvals, bvecs = read_dwi(nifti_file)
        dwi=True
        if convergence is None:
            convergence = '[1000,0.0]'
        if bspline is None:
            bspline = '[150,3]'
    except:
        dwi=False
        # Keep convergence and bspline as None to use defaults 
     
    try:
        if dwi:
            # extract b0 
            logging.info('Diffusion MRI input detected')
            b0_im = extract_b0(dwi_im, bvals)
            # b0_im.to_filename(os.path.join(tmpdir, 'b0.nii'))
            cmd = [ 'N4BiasFieldCorrection', '-d', '3', '-i', os.path.join(tmpdir, 'b0.nii'),
                    '-r', '-b', bspline,'-c', convergence, '-v', '1',
                    '-o', '[{0},{1}]'.format(os.path.join(tmpdir, 'output.nii'), os.path.join(tmpdir, 'field.nii'))
            ]
            # Winsorize the B0 image and build command 
            b0_data = b0_im.get_fdata()
            if mask_nii:
                cmd += ['-x',mask_nii]
                b0_data *= (read_nifti(mask_nii).get_fdata() > 0)
            if weight_nii:
                cmd += ['-w',weight_nii]
                b0_data *= (read_nifti(weight_nii).get_fdata() > 0)
            logging.info('Winsorizing B0 data')
            percLow = percentile(b0_data[b0_data>0],0.5)
            percHigh = percentile(b0_data[b0_data>0],99.5)
            b0_winsor = b0_im.get_fdata().copy()
            b0_winsor[b0_winsor < percLow] = percLow
            b0_winsor[b0_winsor < percHigh] = percHigh
            nifti_image(b0_winsor, b0_im.affine).to_filename(os.path.join(tmpdir, 'b0.nii'))
            
            ExecCommand(cmd).run()
            output_im = read_nifti(os.path.join(tmpdir, 'output.nii'))
            field_im = read_nifti(os.path.join(tmpdir, 'field.nii'))
            # Correct each volume of the dwi 
            dwi_data = dwi_im.get_fdata()
            field_data = field_im.get_fdata()
            logging.info('Divide DWI image every volume by bias field map')
            dwi_corr = dwi_data / field_data[...,None]
            dwi_corr_im = nifti_image(dwi_corr, dwi_im.affine)
            logging.info('Write DWI output')
            write_dwi(output, dwi_corr_im, bvals, bvecs)
            if field:
                logging.info('Write bias field output')
                write_nifti(field, nifti_image(field_data, field_im.affine))
        else:
            cmd = [ 'N4BiasFieldCorrection', '-d', '3', '-i', nifti_file, '-v', '1',
                    '-o', '[{0},{1}]'.format(os.path.join(tmpdir, 'output.nii'), os.path.join(tmpdir, 'field.nii'))
            ]
            if bspline:
                cmd += ['-b', bspline]
            if convergence:
                cmd += ['-c', convergence]
            if mask_nii:
                cmd += ['-x',mask_nii]
            if weight_nii:
                cmd += ['-w',weight_nii]
            ExecCommand(cmd).run()
            logging.info('Write bias corrected output')
            output_im = read_nifti(os.path.join(tmpdir, 'output.nii'))
            field_im = read_nifti(os.path.join(tmpdir, 'field.nii'))
            write_nifti(output, nifti_image(output_im.get_fdata(), output_im.affine))
            if field:
                logging.info('Write bias field output')
                write_nifti(field, nifti_image(field_im.get_fdata(), field_im.affine))
    finally:            
        logging.debug("Remove temporary directory")
        shutil.rmtree(tmpdir)
        logging.info("Done")
    
if __name__ == '__main__': 
    main(sys.argv[1:])
