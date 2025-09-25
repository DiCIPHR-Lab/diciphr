#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_dwi, json_files_from_niftis
from diciphr.diffusion import round_bvals, run_topup, most_gradients_pe, prepare_acqparams_json, prepare_acqparams_nojson

DESCRIPTION = '''
    Run topup on DWI files.
'''

PROTOCOL_NAME='Run_Topup'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-d', action='store', metavar='dwi', dest='dwi_files',
                    type=str, required=True, nargs="*",
                    help='The DWI and reverse phase-encoding/topup filename(s) in Nifti format, separated by a space.'
                    )
    p.add_argument('-o', action='store', metavar='output_base', dest='output_base',
                    type=str, required=True, 
                    help='Output prefix for topup.'
                    )
    p.add_argument('-b', action='store', metavar='bvals', dest='bval_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The bvals files, if not given will strip Nifti extension off the DWI images to ascertain filenames.'
                    )
    p.add_argument('-r', action='store', metavar='bvecs', dest='bvec_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The bvecs files, if not given will strip Nifti extension off the DWI imagse to ascertain filenames.'
                    )
    p.add_argument('-j', action='store', metavar='json', dest='json_files',
                    type=str, required=False, default=[], nargs="*",
                    help='The json files, for Siemens, containing PhaseEncodingDirection, EffectiveEchoSpacing, SliceTiming'
                    )       
    p.add_argument('-m', action='store', metavar='mbfactor', dest='mbfactor',
                    type=int, required=False, default=None, 
                    help='The multi-band factor, if using .json file with SliceTiming entry to determine slspec.txt'
                    )                     
    p.add_argument('-t', action='store', metavar='float', dest='readout_time',
                    type=float, required=False, default=0.062, 
                    help='The readout time, if not using json file to determine it'
                    )        
    p.add_argument('-p', action='store', metavar='pe_dir', dest='phase_enc',
                    type=str, required=False, default=[], nargs="*", 
                    help='The phase enconding direction, if not using json file to determine it. Either LR, RL, AP, PA, IS, SI'
                    )                          
    p.add_argument('-S', action='store', metavar='slspec', dest='slspec',
                    type=str, required=False, default=None,
                    help='The slice timing text file to pass to topup, if not using json file to determine it'
                    )     
    p.add_argument('-c', action='store', metavar='config', dest='config',
                    type=str, required=False, default=None,
                    help='The config file to pass to topup. Default will use the FSL default b02b0.cnf file.'
                    )     
    p.add_argument('-A', '--all-pes', action='store_true', dest='keep_all_pes',
                    help='If provided, will concatenate all phase encoding dirs into final DWI image. ' + 
                    'Use for data acquired with full sequences repeated with opposing phase encoding dirs. ' + 
                    'Default will keep the phase encoding direction with the most number of weighted volumes.'
                    )
    return p

def unique_acqparams(acqparams_list):
    uniq = set()
    for acqparams_line in acqparams_list:
        acq = acqparams_line[:3]
        uniq.add(tuple(acq))
    return len(uniq)
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output_base))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
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
        else:
            args.json_files = json_files_from_niftis(args.dwi_files)
        if args.phase_enc:
            if len(args.phase_enc) != nimgs:
                raise ValueError('Number of phase encoding entries must match number of DWI files')
                
        # Read DWI data and separate into images, bvals and bvecs 
        if args.bval_files:
            dwis_read = [read_dwi(*tup) for tup in zip(args.dwi_files, args.bval_files, args.bvec_files)]
        else:
            dwis_read = [read_dwi(dwifn, force=True) for dwifn in args.dwi_files]
        # convert list of [(dwi,bval,bvec)] tuples into list of dwis, list of bvals, list of bvecs 
        dwis, bvals, bvecs = map(list, tuple(zip(*dwis_read)))
        bvals = [round_bvals(bv) for bv in bvals]
        
        # Get acquisition parameters from json files or from command line 
        if args.json_files:
            logging.info("Get acquisition parameters from .json files")
            all_acqparams = [prepare_acqparams_json(fn, dwi_im) for fn, dwi_im in zip(args.json_files, dwis)]
        else:
            logging.info("Get acquisition parameters without .json files")
            all_acqparams = [prepare_acqparams_nojson(args.readout_time, phase_enc) for phase_enc in args.phase_encs]
        
        if unique_acqparams(all_acqparams) <= 1:
            raise ValueError('Multiple phase encoding directions not detected')
        
        # Array of which DWI images to keep in output 
        if args.keep_all_pes:
            keep_dwis = [ True for dwi in dwis ]
        else:
            keep_dwis = most_gradients_pe(bvals, all_acqparams)
        if not all(keep_dwis):
            topup_phase_enc = [p for p,k in zip(args.phase_encs, keep_dwis) if not k][0]
            logging.info(f"Diffusion images with phase-encoding direction {topup_phase_enc} used only for topup")
            
        # run topup
        logging.info("Run topup")
        run_topup(dwis, bvals, bvecs, all_acqparams, args.output_base, keep_dwis=keep_dwis, config=args.config)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
if __name__ == '__main__': 
    main(sys.argv[1:])
