#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import read_nifti
from diciphr.registration import ants_registration_dti_t1

DESCRIPTION = '''
    Runs ANTs Registration
'''

PROTOCOL_NAME='Ants_Registration'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-m', '-mode', action='store', metavar='<str>', dest='mode', 
                    type=str, required=True, 
                    help='Mode of operation: Options: DTI-T1, T1-Eve, DTI-Eve, T1-MNI'
                    )
    p.add_argument('-s', action='store', metavar='<str>', dest='subject',
                    type=str, required=True, 
                    help='Subject'
                    )
    p.add_argument('-o', action='store', metavar='<dir>', dest='outdir',
                    type=str, required=True, 
                    help='Output directory'
                    )
    p.add_argument('-t', action='store', metavar='<nii>', dest='t1_file',
                    type=str, required=True, 
                    help='Input skull-stripped T1 filename'
                    )
    p.add_argument('-b', action='store', metavar='<nii>', dest='b0_file',
                    type=str, required=False, default=None, 
                    help='Input B0 filename'
                    )
    p.add_argument('-f', action='store', metavar='<nii>', dest='fa_file',
                    type=str, required=False, default=None,  
                    help='Input FA filename'
                    )
    p.add_argument('-x', action='store', metavar='<nii>', dest='mask_file',
                    type=str, required=False, 
                    help='Input DTI-space mask filename'
                    )
    p.add_argument('-P', '--phaseenc', action='store', metavar='<str>', dest='phase_enc',
                    type=str, required=False, default=None, 
                    help='The phase encoding direction, this option enables SyN DTI-T1 registration and is used to restrict the SyN deformation'
                    )
#    p.add_argument('-i', action='store', metavar='<str>', dest='initialize',
#                    type=str, required=False, 
#                    help='Registration initialization method. Options: antsAI (default for DTI-T1), identity, origin, centroid, a .txt (ITK-Snap) or .mat (ANTs) affine transformation file'
#                    )
#    p.add_argument('-T', '--transform-type', action='store', metavar='<str>', dest='transform_type',
#                    type=str, required=False, default='r', 
#                    help='The transform type. Options: r (rigid), a (2 stage affine), s (3 stage fully deformable SyN), rs (restricted SyN, for DTI-T1 method)' 
#                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    mode = args.mode.upper()
    if mode not in ('DTI-T1', 'T1-Eve', 'T1-MNI', 'DTI-Eve'):
        raise ValueError('Invalid mode provided')
    make_dir(args.outdir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        if mode.upper() == 'DTI-T1':
            registration_dti_t1(args)
        else:
            raise ValueError('Mode not implemented yet')
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def registration_dti_t1(args):
    check_inputs(args.t1_file, nifti=True)
    t1_img = read_nifti(args.t1_file)
    check_inputs(args.outdir, directory=True)
    output_prefix = os.path.join(args.outdir, args.subject)
    if args.phase_enc is None:
        syn = False
    elif args.phase_enc.upper() in ('AP','PA','LR','RL','IS','SI'):
        syn = True
    else:
        raise ValueError("Phase encoding direction must be one of ('AP','PA','LR','RL','IS','SI')")
    check_inputs(args.b0_file, nifti=True)    
    b0_img = read_nifti(args.b0_file)
    if args.fa_file:
        check_inputs(args.fa_file, nifti=True)
        fa_img = read_nifti(args.fa_file)
    else:
        fa_img = None
    if args.mask_file:
        check_inputs(args.mask_file, nifti=True)
        dti_mask_img = read_nifti(args.mask_file)
    else:
        dti_mask_img = None
    syn = args.phase_enc is not None 
    ants_registration_dti_t1(output_prefix, b0_img, t1_img, fa_img=fa_img, dti_mask_img=dti_mask_img, syn=syn, phase_enc=args.phase_enc)

if __name__ == '__main__': 
    main(sys.argv[1:])
