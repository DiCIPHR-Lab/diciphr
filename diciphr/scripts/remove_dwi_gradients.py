#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrException, DiciphrArgumentParser
from diciphr.nifti_utils import ( read_nifti, write_nifti, read_dwi, write_dwi, 
                reorient_dwi, reorient_nifti, is_valid_dwi )
from diciphr.diffusion import remove_dwi_gradients
import nibabel as nib

DESCRIPTION = '''
    Remove gradient images from a DWI volume. 
'''

PROTOCOL_NAME='Remove_DWI_Gradients'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwifile',dest='dwifile',
                    type=str, required=True, 
                    help='Input DWI filename'
                    )
    p.add_argument('-o',action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='Output filename'
                    )
    p.add_argument('-x',action='store',metavar='list',dest='gradients',
                    type=str, required=True, default='', 
                    help='Gradients to remove, separated by commas'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        output_dir = os.path.dirname(os.path.realpath(args.output))
        make_dir(output_dir, recursive=True, pass_if_exists=True)
        dwifile = check_inputs(args.dwifile, nifti=True)
        gradients = list(map(int,args.gradients.split(',')))
        check_inputs(output_dir, directory=True)
        run_remove_dwi_gradients(dwifile, args.output, gradients)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_remove_dwi_gradients(dwifile, output, gradients_to_remove):
    ''' 
    Remove DWI gradients 
    '''
    logging.info('DWI: {}'.format(dwifile))
    logging.info('Output file: {}'.format(output))
    logging.info('Remove: {}'.format(gradients_to_remove))
    
    logging.info('Begin Protocol {}'.format(PROTOCOL_NAME))    
    # Load dwifile
    logging.info('Read input nifti')
    
    dwi_im, bvals, bvecs = read_dwi(dwifile)
    out_dwi_im, out_bvals, out_bvecs = remove_dwi_gradients(dwi_im, bvals, bvecs, gradients_to_remove)
    is_valid_dwi(out_dwi_im, out_bvals, out_bvecs, True)
    write_dwi(output, out_dwi_im, out_bvals, out_bvecs)
    
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
