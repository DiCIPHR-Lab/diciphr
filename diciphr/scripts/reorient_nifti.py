#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser, DiciphrException
from diciphr.nifti_utils import read_nifti, write_nifti, read_dwi, write_dwi, reorient_dwi, reorient_nifti
import nibabel as nib

DESCRIPTION = '''
    Reorient a Nifti volume to lab-default LPS orientation. 
'''

PROTOCOL_NAME='Reorient_Nifti'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', '-d', action='store',metavar='datafile',dest='datafile',
                    type=str, required=True, 
                    help='Input filename'
                    )
    p.add_argument('-o',action='store',metavar='outputfile',dest='outputfile',
                    type=str, required=True, 
                    help='Output filename'
                    )
    p.add_argument('-r',action='store',metavar='orn_string',dest='orn_string',
                    type=str, required=False, default='LPS', 
                    help='Orientation string. Default LPS'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.outputfile))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.datafile, nifti=True)
        check_inputs(output_dir, directory=True)
        run_reorient_nifti(args.datafile, args.outputfile, orn_string=args.orn_string)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_reorient_nifti(datafile, outputfile, orn_string='LPS'):
    ''' 
    Run the DTI Preprocessing protocol.
    
    Parameters
    ----------
    datafile : str
        Probtrackx directory.
    outputfile : str
        Target labels file from freesurfer_postprocess
    orn_string : Optional[str]
        Orientation string of 
    Returns
    -------
    None
    '''
    logging.info('datafile: {}'.format(datafile))
    logging.info('outputfile: {}'.format(outputfile))
    logging.info('orn_string: {}'.format(orn_string))
    
    logging.info('Begin Protocol {}'.format(PROTOCOL_NAME))    
    # Load datafile
    logging.info('Read input nifti')
    diffusion=False
    try:
        dwi_im, bvals, bvecs = read_dwi(datafile)
        diffusion=True
        logging.info('Diffusion volume detected.')
    except:
        nifti_im = read_nifti(datafile)

    # Output filenames 
    if diffusion:
        logging.info('Reorienting diffusion volume.')
        dwi_reor_im, bvals_reor, bvecs_reor = reorient_dwi(dwi_im, bvals, bvecs, orientation=orn_string)
        logging.info('Saving to file {}.'.format(outputfile))
        write_dwi(outputfile, dwi_reor_im, bvals_reor, bvecs_reor) 
    else:
        logging.info('Reorienting Nifti volume.')
        nifti_reor_im = reorient_nifti(nifti_im, orientation=orn_string)
        logging.info('Saving to file {}.'.format(outputfile))
        write_nifti(outputfile, nifti_reor_im)
    
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
