#! /usr/bin/env python

import os, sys, argparse, logging, traceback, shutil
from ..utils import ( check_inputs, make_dir, 
                protocol_logging, DiciphrException, ExecCommand )
from ..nifti_utils import read_nifti, read_dwi, write_dwi, nifti_image, concatenate_niftis
from ..diffusion import ( round_bvals, extract_b0, concatenate_dwis, run_topup, 
                decode_phaseenc, prepare_acqparams_json, prepare_acqparams_nojson, prepare_index )
import nibabel as nib
import numpy as np

DESCRIPTION = '''
    Run topup on DWI files.
'''

PROTOCOL_NAME='Run_Topup'    
    
def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-d',action='store',metavar='dwi',dest='dwi_files',
                    type=str, required=True, nargs="*",
                    help='The DWI and reverse phase-encoding/topup filename(s) in Nifti format, separated by a space.'
                    )
    p.add_argument('-o',action='store',metavar='output_base',dest='output_base',
                    type=str, required=True, 
                    help='Output prefix for topup.'
                    )
    p.add_argument('-b',action='store',metavar='bvals',dest='bval_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The bvals files, if not given will strip Nifti extension off the DWI images to ascertain filenames.'
                    )
    p.add_argument('-r',action='store',metavar='bvecs',dest='bvec_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The bvecs files, if not given will strip Nifti extension off the DWI imagse to ascertain filenames.'
                    )
    p.add_argument('-j',action='store',metavar='json',dest='json_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The json files, for Siemens, containing PhaseEncodingDirection, EffectiveEchoSpacing, SliceTiming'
                    )       
    p.add_argument('-m',action='store',metavar='mbfactor',dest='mbfactor',
                    type=int, required=False, default=None, 
                    help='The multi-band factor, if using .json file with SliceTiming entry to determine slspec.txt'
                    )                     
    p.add_argument('-t',action='store',metavar='float',dest='readout_time',
                    type=float, required=False, default=0.062, 
                    help='The readout time, if not using json file to determine it'
                    )        
    p.add_argument('-p',action='store',metavar='pe_dir',dest='phase_enc',
                    type=str, required=False, default=[], nargs="*", 
                    help='The phase enconding direction, if not using json file to determine it. Either LR, RL, AP, PA, IS, SI'
                    )                          
    p.add_argument('-S',action='store',metavar='slspec', dest='slspec',
                    type=str, required=False, default=None,
                    help='The slice timing text file to pass to topup, if not using json file to determine it'
                    )     
    p.add_argument('-c',action='store',metavar='config', dest='config',
                    type=str, required=False, default='b02b0.cnf',
                    help='The config file to pass to topup. Default will use the FSL default b02b0.cnf file.'
                    )     
    p.add_argument('--concatenate', action='store_true', dest='concatenate', 
                    required=False, default=False, 
                    help='Concatenate DWIs if provided (e.g. for data acquired half LR, half RL), else, will perform topup with all data but return DWI and index files corresponding to first DWI only (e.g for data acquired full DWI followed by TOPUP scan of B0s)'
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
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        check_inputs(*args.dwi_files, nifti=True)
        nimgs = len(args.dwi_files)
        if args.bval_files:
            check_inputs(*args.bval_files)
            if len(args.bval_files) != nimgs:
                raise ValueError('Number of bval files must match number of DWI files')
        if args.bvec_files:
            check_inputs(*args.bvec_files)
            if len(args.bvec_files) != nimgs:
                raise ValueError('Number of bvec files must match number of DWI files')
        if args.json_files:
            check_inputs(*args.json_files)
            if len(args.json_files) != nimgs:
                raise ValueError('Number of json files must match number of DWI files')
        if args.phase_enc:
            if len(args.phase_enc) != nimgs:
                raise ValueError('Number of phase encoding entries must match number of DWI files')
                
        dwis = []
        bvals = []
        bvecs = [] 
        for i in range(len(args.dwi_files)):
            if args.bval_files:
                dwi, bval, bvec = read_dwi(args.dwi_files[i], args.bval_files[i], args.bvec_files[i])
            else:
                dwi, bval, bvec = read_dwi(args.dwi_files[i])
            dwis.append(dwi)
            bvals.append(bval)
            bvecs.append(bvec)
        run_topup(dwis, bvals, bvecs, args.output_base, args.json_files, 
                phase_encs=args.phase_enc, readout_time=args.readout_time, 
                mbfactor=args.mbfactor, slspec=args.slspec, 
                concatenate=args.concatenate, config=args.config)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    
if __name__ == '__main__': 
    main(sys.argv[1:])
