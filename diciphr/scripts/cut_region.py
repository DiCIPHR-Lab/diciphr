#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from diciphr.utils import ( check_inputs, make_dir, 
                protocol_logging, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, write_nifti, 
                cut_region )

DESCRIPTION = '''
    Cut a Nifti ROI into multiple regions. 
'''

PROTOCOL_NAME='cut_region'
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-i',action='store',metavar='roifile',dest='roifile',
                    type=str, required=True, 
                    help='Input ROI filename'
                    )
    p.add_argument('-o',action='store',metavar='outbase',dest='outbase',
                    type=str, required=True,
                    help='Output filebase. Will be appended with 001 002 etc.'
                    )
    p.add_argument('-k', action='store', metavar='k', dest='k',
                    type=int, required=True, 
                    help='X adjustment, two integers separated by a space. Negative values crop, positive values pad.'
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
    output_dir = os.path.dirname(os.path.realpath(args.outbase))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(args.roifile, nifti=True)
        check_inputs(output_dir, directory=True)
        sub_images = cut_region(read_nifti(args.roifile), args.k)
        for i, im in enumerate(sub_images):
            filename_out = '{0}_{1:03d}.nii.gz'.format(args.outbase, i)
            write_nifti(filename_out, im)         
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
if __name__ == '__main__': 
    main(sys.argv[1:])
