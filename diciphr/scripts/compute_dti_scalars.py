#! /usr/bin/env python

import os, sys, shutil, logging, traceback, argparse
from ..utils import ( check_inputs, make_dir, 
                protocol_logging, DiciphrException )
from ..nifti_utils import ( read_nifti, write_nifti, 
                strip_nifti_ext, get_nifti_ext )
from ..diffusion import TensorScalarCalculator, is_tensor

DESCRIPTION = '''
    Calculates diffusion scalar maps from a tensor image. 
'''

PROTOCOL_NAME='Compute_DTI_Scalars'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', action='store', metavar='tensor_filename', dest='tensor_filename',
                    type=str, required=True,
                    help='path of the DTI tensor file'
                    )
    p.add_argument('-o', action='store', metavar='output_dir', dest='output_dir',
                    type=str, required=False, default=None,
                    help='Name of output directory. Defaults to directory of input file'
                    )
    p.add_argument('-m', action='store', metavar='mask_filename', dest='mask_filename', 
                    type=str, required=False, default=None, 
                    help='Provide a brain mask nifti file')
    p.add_argument('-f', action='store_true', dest='calculate_fa', 
                    required=False, default=False, 
                    help='calculate and save an fractional anisotropy map (FA)')
    p.add_argument('-t', action='store_true', dest='calculate_tr', 
                    required=False, default=False, 
                    help='calculate and save a trace map (TR)')
    p.add_argument('-y', action='store_true', dest='calculate_md', 
                    required=False, default=False, 
                    help='calculate and save a mean diffusivity map (MD)')                
    p.add_argument('-x', action='store_true', dest='calculate_axrad', 
                    required=False, default=False, 
                    help='calculate and save radial and axial diffusivity (AX, RAD)')
    p.add_argument('-g', action='store_true', dest='calculate_geo', 
                    required=False, default=False, 
                    help='calculate and save geometric features (CL, CP, CS)')
    p.add_argument('-e', action='store_true', dest='calculate_eigvals', 
                    required=False, default=False, 
                    help='calculate and save eigenvalues')
    p.add_argument('-v', action='store_true', dest='calculate_eigvecs', 
                    required=False, default=False, 
                    help='calculate and save eigenvectors')
    p.add_argument('-c', action='store_true', dest='calculate_colormap',
                    required=False, default=False, 
                    help='calculate FA weighted RGB colormap')
    p.add_argument('--all', action='store_true', dest='calculate_all', 
                    required=False, default=False, 
                    help='calculate all the images')
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
    if args.output_dir is None:
        args.output_dir = os.path.realpath(os.path.dirname(args.tensor_filename))
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(args.tensor_filename, nifti=True)
        if not is_tensor(read_nifti(args.tensor_filename)):
            raise DiciphrException('Input image is not tensor!')
        if args.mask_filename is not None:
            check_inputs(args.mask_filename, nifti=True)
        # Decide what to do 
        if args.calculate_all:
            tasks=['f','t','m','x','g','e','v','c']
        else:
            tasks=[]
            if args.calculate_fa:
                tasks.append('f')
            if args.calculate_tr:
                tasks.append('t')
            if args.calculate_md:
                tasks.append('m')
            if args.calculate_axrad:
                tasks.append('x')
            if args.calculate_geo:
                tasks.append('g')
            if args.calculate_eigvals:
                tasks.append('e')
            if args.calculate_eigvecs:
                tasks.append('v')
            if args.calculate_colormap:
                tasks.append('c')
        tasks = list(set(tasks)) # pare down to unique entries
        if len(tasks) < 1:
            raise DiciphrException('Nothing to do!') 
        run_diffusion_scalar_calculator(args.tensor_filename, args.output_dir, tasks, mask_filename=args.mask_filename)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
def run_diffusion_scalar_calculator(tensor_filename, output_dir, tasks, mask_filename=None):
    ''' 
    Calculate DTI Scalars from a tensor file
    
    Parameters
    ----------
    tensor_filename : str 
        Tensor file
    output_dir : str
        Output directory.
    tasks : list
        List of scalars to calculate. Acceptable elements are ['f','t','x','g','e']
    mask_filename : Optional[str]
        If provided, scalar maps will be calculated within this mask.     
    Returns
    -------
    None
    '''
    calculate_fa = 'f' in tasks
    calculate_tr = 't' in tasks
    calculate_axrad = 'x' in tasks
    calculate_geo = 'g' in tasks
    calculate_eigvals = 'e' in tasks
    calculate_eigvecs = 'v' in tasks
    calculate_colormap = 'c' in tasks
    calculate_md = 'm' in tasks
    logging.info('tensor_filename: {}'.format(tensor_filename))
    logging.info('output_dir: {}'.format(output_dir))
    if mask_filename:
        logging.info('mask_filename: {}'.format(mask_filename))
    if calculate_fa:
        logging.info('Will calculate FA')
    if calculate_tr:
        logging.info('Will calculate TR')
    if calculate_md:
        logging.info('Will calculate MD')
    if calculate_axrad:
        logging.info('Will calculate axial and radial diffusivity')
    if calculate_geo:
        logging.info('Will calculate geometrical features (CL, CP, CS)')
    if calculate_eigvals:
        logging.info('Will calculate eigenvalues')
    if calculate_eigvecs:
        logging.info('Will calculate eigenvectors')
    if calculate_colormap:
        logging.info('Will calculate FA weighted RGB colormap')
    
    logging.info('Begin Protocol')
    
    tensor_filebase = strip_nifti_ext(os.path.basename(tensor_filename))
    tensor_filebase = os.path.join(output_dir, tensor_filebase)
    nifti_ext = get_nifti_ext(tensor_filename)
    
    logging.info('Load tensor data')
    tensor_im = read_nifti(tensor_filename)
    
    mask_im = None
    if mask_filename:
        logging.info('Load mask data')
        mask_im = read_nifti(mask_filename)
    
    logging.info('Initialize TensorScalarCalculator')
    TSC = TensorScalarCalculator(tensor_im, mask_im=mask_im)
    
    if calculate_fa:
        logging.info('Calculating FA')
        fa_im = TSC.FA
        fa_filename = tensor_filebase + '_FA.' + nifti_ext
        logging.info('Writing FA image to file {}'.format(fa_filename))
        write_nifti(fa_filename, fa_im)
    if calculate_tr:
        logging.info('Calculating TR')
        tr_im = TSC.TR
        tr_filename = tensor_filebase + '_TR.' + nifti_ext
        logging.info('Writing TR image to file {}'.format(tr_filename))
        write_nifti(tr_filename, tr_im)
    if calculate_md:
        logging.info('Calculating MD')
        md_im = TSC.MD
        md_filename = tensor_filebase + '_MD.' + nifti_ext
        logging.info('Writing MD image to file {}'.format(md_filename))
        write_nifti(md_filename, md_im)
    if calculate_axrad:
        logging.info('Calculating AX')
        ax_im = TSC.AX
        ax_filename = tensor_filebase + '_AX.' + nifti_ext
        logging.info('Writing AX image to file {}'.format(ax_filename))
        write_nifti(ax_filename, ax_im)
        
        logging.info('Calculating RAD')
        rad_im = TSC.RAD
        rad_filename = tensor_filebase + '_RAD.' + nifti_ext
        logging.info('Writing RAD image to file {}'.format(rad_filename))
        write_nifti(rad_filename, rad_im)
    if calculate_geo:
        logging.info('Calculating CL')
        cl_im = TSC.CL
        cl_filename = tensor_filebase + '_CL.' + nifti_ext
        logging.info('Writing CL image to file {}'.format(cl_filename))
        write_nifti(cl_filename, cl_im)
        
        logging.info('Calculating CP')
        cp_im = TSC.CP
        cp_filename = tensor_filebase + '_CP.' + nifti_ext
        logging.info('Writing CP image to file {}'.format(cp_filename))
        write_nifti(cp_filename, cp_im)
        
        logging.info('Calculating CS')
        cs_im = TSC.CS
        cs_filename = tensor_filebase + '_CS.' + nifti_ext
        logging.info('Writing CS image to file {}'.format(cs_filename))
        write_nifti(cs_filename, cs_im)
    if calculate_eigvals:
        logging.info('Calculating eigenvalues')
        eigen_ims = TSC.eigenvalues  #L1 L2 L3
        for idx in [ 1, 2, 3 ]:
            eig_filename = tensor_filebase + '_L{}.'.format(idx) + nifti_ext
            logging.info('Writing L{0} image to file {1}'.format(idx, eig_filename))
            write_nifti(eig_filename, eigen_ims[idx-1])
    if calculate_eigvecs:
        logging.info('Calculating eigenvectors')
        eigen_ims = TSC.eigenvectors  #L1 L2 L3
        for idx in [ 1, 2, 3 ]:
            eig_filename = tensor_filebase + '_V{}.'.format(idx) + nifti_ext
            logging.info('Writing V{0} image to file {1}'.format(idx, eig_filename))
            write_nifti(eig_filename, eigen_ims[idx-1])
    if calculate_colormap:
        logging.info('Calculating FA weighted RGB colormap')
        color_im = TSC.colormap
        color_filename = tensor_filebase + '_colormap.' + nifti_ext
        logging.info('Writing colormap image to file {}'.format(color_filename))
        write_nifti(color_filename, color_im)
            
    logging.info('End of protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
