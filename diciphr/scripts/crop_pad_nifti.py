#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
                DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, write_nifti, 
                strip_nifti_ext, crop_pad_image )

DESCRIPTION = '''
    Crop or pad a Nifti volume. 
'''

PROTOCOL_NAME='crop_pad_nifti'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='datafile',dest='datafile',
                    type=str, required=True, 
                    help='Input filename'
                    )
    p.add_argument('-o',action='store',metavar='outputfile',dest='outputfile',
                    type=str, required=True,
                    help='Output filename'
                    )
    p.add_argument('-x',action='store',metavar='x_adjust',dest='x_adjust',nargs=2,
                    type=str, required=False, default=[0,0], 
                    help='X adjustment, two integers separated by a space. Negative values crop, positive values pad.'
                    )
    p.add_argument('-y',action='store',metavar='y_adjust',dest='y_adjust',nargs=2,
                    type=str, required=False, default=[0,0], 
                    help='Y adjustment, two integers separated by a space. Negative values crop, positive values pad.'
                    )
    p.add_argument('-z',action='store',metavar='z_adjust',dest='z_adjust',nargs=2,
                    type=str, required=False, default=[0,0], 
                    help='Z adjustment, two integers separated by a space. Negative values crop, positive values pad.'
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
        run_crop_pad_image(args.datafile, args.outputfile, x_adjust=args.x_adjust, y_adjust=args.y_adjust, z_adjust=args.z_adjust)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_crop_pad_image(datafile, outputfile, x_adjust=[0,0], y_adjust=[0,0], z_adjust=[0,0]):
    logging.info('datafile: {}'.format(datafile))
    logging.info('outputfile: {}'.format(outputfile))
    logging.info('x_adjust: {}'.format(x_adjust))
    logging.info('y_adjust: {}'.format(y_adjust))
    logging.info('z_adjust: {}'.format(z_adjust))
    x_adjust=list(map(int,x_adjust))
    y_adjust=list(map(int,y_adjust))
    z_adjust=list(map(int,z_adjust))
    try:
        im, bvals, bvecs = read_dwi(datafile)
        dwi=True 
    except:
        im = read_nifti(datafile)
        dwi=False 
    crop_im = crop_pad_image(im, x_adjust=x_adjust, y_adjust=y_adjust, z_adjust=z_adjust)    
    if dwi:
        logging.info(f'Write DWI nifti file {outputfile}')
        write_dwi(outputfile, crop_im, bvals, bvecs)
    else:
        logging.info(f'Write nifti file {outputfile}')
        write_nifti(outputfile, crop_im)
    
if __name__ == '__main__': 
    main(sys.argv[1:])
