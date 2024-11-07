#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from ..utils import ( check_inputs, make_dir, 
                protocol_logging, DiciphrException )
from ..nifti_utils import ( read_nifti, write_nifti, 
                strip_nifti_ext, crop_pad_image )

DESCRIPTION = '''
    Crop or pad a Nifti volume. 
'''

PROTOCOL_NAME='crop_pad_nifti'    
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
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
    output_dir = os.path.dirname(os.path.realpath(args.outputfile))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(args.datafile, nifti=True)
        check_inputs(output_dir, directory=True)
        run_crop_pad_image(args.datafile, args.outputfile, x_adjust=args.x_adjust, y_adjust=args.y_adjust, z_adjust=args.z_adjust)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
def run_crop_pad_image(datafile, outputfile, x_adjust=[0,0], y_adjust=[0,0], z_adjust=[0,0]):
    logging.info('datafile: {}'.format(datafile))
    logging.info('outputfile: {}'.format(outputfile))
    logging.info('x_adjust: {}'.format(x_adjust))
    logging.info('y_adjust: {}'.format(y_adjust))
    logging.info('z_adjust: {}'.format(z_adjust))
    x_adjust=list(map(int,x_adjust))
    y_adjust=list(map(int,y_adjust))
    z_adjust=list(map(int,z_adjust))
    im = read_nifti(datafile)
    crop_im = crop_pad_image(im, x_adjust=x_adjust, y_adjust=y_adjust, z_adjust=z_adjust)    
    write_nifti(outputfile, crop_im)
    #copy bval/bvec if exist
    if os.path.exists(strip_nifti_ext(datafile)+'.bval'):
        logging.info('Copying bval and bvec files')
        shutil.copyfile(strip_nifti_ext(datafile)+'.bval', strip_nifti_ext(outputfile)+'.bval')
        shutil.copyfile(strip_nifti_ext(datafile)+'.bvec', strip_nifti_ext(outputfile)+'.bvec')
    
if __name__ == '__main__': 
    main(sys.argv[1:])
