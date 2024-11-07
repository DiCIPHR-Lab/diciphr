#! /usr/bin/env python 
import os, sys, logging, traceback, argparse, time
from ..fernet.pipeline import fernet_regions 
from ..utils import check_inputs, make_dir, protocol_logging
    
DESCRIPTION = '''
Creates WM and CSF rois for running FERNET. 
'''
PROTOCOL_NAME = 'FERNET_regions' 

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
    p.add_argument('-x','--exclude',action='store',metavar='mask',dest='exclude_mask',
                    type=str, required=False, default=None, 
                    help='A mask (e.g. peritumoral region) of voxels to exclude when getting typical WM, GM voxels.'
                    )
    p.add_argument('-o', '--output', action='store', metavar='output', dest='output',
                    type=str, required=True,
                    help='Output basename for rois.'
                    )
    p.add_argument('-f','--fa-threshold',action='store',metavar='fa',dest='fa_threshold',
                    type=float, required=False, default=0.7, 
                    help='The FA threshold to define the WM roi. Default is 0.7'
                    )
    p.add_argument('-t','--tr-threshold',action='store',metavar='tr',dest='tr_threshold',
                    type=float, required=False, default=0.0085, 
                    help='The TR threshold to define the CSF roi. Default is 0.0085'
                    )
    p.add_argument('-n','--erode-iters',action='store',metavar='int',dest='erode_iterations',
                    type=int, required=False, default=8, 
                    help='Erode the mask this many iterations to narrow in on ventricles and deep WM. Default is 8'
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
        if args.exclude_mask:
            check_inputs(args.exclude_mask, nifti=True)
        fernet_regions(args.dwi, args.bvals, args.bvecs, args.mask, args.output, exclude_mask=args.exclude_mask, 
            fa_threshold=args.fa_threshold, tr_threshold=args.tr_threshold, erode_iterations=args.erode_iterations)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
if __name__ == '__main__':
    main(sys.argv[1:])
