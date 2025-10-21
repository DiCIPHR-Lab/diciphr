#! /usr/bin/env python

import os, sys, shutil, logging
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import ( read_nifti, write_nifti, read_dwi, write_dwi, 
                                 strip_nifti_ext, reorient_dwi, reorient_nifti )

DESCRIPTION = '''
    Reorient a Nifti volume to lab-default LPS orientation. 
'''

PROTOCOL_NAME='Reorient_Nifti'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', '-d', action='store', metavar='datafile', dest='datafile',
                    type=str, required=True, 
                    help='Input filename'
                    )
    p.add_argument('-o', action='store', metavar='outputfile', dest='outputfile',
                    type=str, required=True, 
                    help='Output filename'
                    )
    p.add_argument('-r', action='store', metavar='orn_string', dest='orn_string',
                    type=str, required=False, default='LPS', 
                    help='Orientation string. Default LPS'
                    )
    p.add_argument('-j', '--copy-json', action='store_true', dest='copy_json', 
                    required=False, 
                    help='Copy any associated .json file to output'
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
        run_reorient_nifti(args.datafile, args.outputfile, orn_string=args.orn_string, copy_json=args.copy_json)
    except Exception:
        logging.exception(f'Exception encountered running {PROTOCOL_NAME}')
        raise
    
def run_reorient_nifti(datafile, outputfile, orn_string='LPS', copy_json=False):
    ''' 
    Run the DTI Preprocessing protocol.
    
    Parameters
    ----------
    datafile : str
        Probtrackx directory.
    outputfile : str
        Target labels file from freesurfer_postprocess
    orn_string : Optional[str]
        Orientation string 
    copy_json : Optional[bool]
        Copy associated .json file to output
    Returns
    -------
    None
    '''
    logging.info(f'datafile: {datafile}')
    logging.info(f'outputfile: {outputfile}')
    logging.info(f'orn_string: {orn_string}')
    
    logging.info(f'Begin Protocol {PROTOCOL_NAME}')
    # Load datafile
    logging.info('Read input nifti')
    diffusion=False
    try:
        dwi_im, bvals, bvecs = read_dwi(datafile)
        diffusion=True
        logging.info('Diffusion volume detected')
    except:
        nifti_im = read_nifti(datafile)

    # Output filenames 
    if diffusion:
        logging.info('Reorienting diffusion volume')
        dwi_reor_im, bvals_reor, bvecs_reor = reorient_dwi(dwi_im, bvals, bvecs, orientation=orn_string)
        logging.info(f'Saving to file {outputfile}')
        write_dwi(outputfile, dwi_reor_im, bvals_reor, bvecs_reor) 
    else:
        logging.info('Reorienting Nifti volume')
        nifti_reor_im = reorient_nifti(nifti_im, orientation=orn_string)
        logging.info(f'Saving to file {outputfile}')
        write_nifti(outputfile, nifti_reor_im)
    
    if copy_json:
        logging.info('Copy associated .json files')
        jsonfile = strip_nifti_ext(datafile)+'.json'
        target_jsonfile = strip_nifti_ext(outputfile)+'.json'
        if os.path.exists(jsonfile):
            shutil.copyfile(jsonfile, target_jsonfile)
        else:
            logging.warning(f'JSON file does not exist: {jsonfile}')
    
    logging.info(f'End of Protocol {PROTOCOL_NAME}')
    
if __name__ == '__main__': 
    main(sys.argv[1:])
