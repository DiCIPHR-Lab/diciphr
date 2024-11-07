#! /usr/bin/env python
import os, sys, logging, traceback, argparse, time
from ..fernet.pipeline import fernet_correct_dwi 
from ..utils import check_inputs, make_dir, protocol_logging

DESCRIPTION = '''
    Apply free water elimination to a DWI image using an existing water volume fraction (VF) map. 
'''

PROTOCOL_NAME = 'FERNET_FW_DWI' 
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d','-k','--data',action='store',metavar='dwi',dest='dwi',
                    type=str, required=True, 
                    help='Input DWIs data file (Nifti or Analyze format).'
                    )
    p.add_argument('-r','--bvecs', action='store', metavar='bvecs', dest='bvecs',
                    type=str, required=False, default=None,
                    help='Gradient directions (.bvec file).'
                    )
    p.add_argument('-b','--bvals', action='store', metavar='bvals', dest='bvals',
                    type=str, required=False, default=None,
                    help='B-values (.bval file).'
                    )
    p.add_argument('-m','--mask',action='store',metavar='mask', dest='mask',
                    type=str, required=True, 
                    help='Brain mask file (Nifti or Analyze format).'
                    )
    p.add_argument('-f','--fraction',action='store',metavar='vf',dest='volume_fraction',
                    type=str, required=False, default=None, 
                    help='The free water volume fraction (VF) map.'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for corrected B0 map and corrected DWI.'
                    )
    p.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    p.add_argument('--logfile', action='store', metavar='log', dest='logfile', 
                    type=str, required=False, default=None, 
                    help='A log file. If not provided will print to stderr. If a dir is provided will create a logfile'
                    )
    return p

def main(argv):    
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir,recursive=True,pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(args.dwi, nifti=True)
        check_inputs(args.mask, nifti=True)
        check_inputs(args.volume_fraction, nifti=True)
        fernet_correct_dwi(args.dwi, args.bvals, args.bvecs, args.mask, args.volume_fraction, args.output)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e

if __name__ == '__main__':
    main(sys.argv[1:])