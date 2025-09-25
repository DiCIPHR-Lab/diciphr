#! /usr/bin/env python

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging,
                    DiciphrArgumentParser, DiciphrException )
import numpy as np

DESCRIPTION = '''
    Flip bvecs file along X, Y, or Z axis.  
'''

PROTOCOL_NAME='Flip_Bvecs'
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i',action='store',dest='bvecfile',
                    type=str, required=True, 
                    help='The input .bvec file.'
                    )
    p.add_argument('-o',action='store',dest='outfile',
                    type=str, required=True, 
                    help='The output .bvec file.'
                    )
    p.add_argument('-x',action='store_true',dest='flip_x',
                    required=False, default=False,
                    help='Flip the bvec along the X direction.'
                    )
    p.add_argument('-y',action='store_true',dest='flip_y',
                    required=False, default=False,
                    help='Flip the bvec along the Y direction.'
                    )
    p.add_argument('-z',action='store_true',dest='flip_z',
                    required=False, default=False,
                    help='Flip the bvec along the Z direction.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.outfile))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    check_inputs(args.bvecfile)
    try:    
        run_flip_bvec(args.bvecfile, args.outfile, 
            args.flip_x, args.flip_y, args.flip_z)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def run_flip_bvec(bvecfile, outfile, flip_x, flip_y, flip_z):
    ''' 
    Flip bvec file along axes. 
    
    Parameters
    ----------
    bvecfile : str
        Input bvec file
    outfile : str
        Output bvec file
    flip_x : bool
        If true, flip along X axis. 
    flip_y : bool
        If true, flip along Y axis. 
    flip_z : bool
        If true, flip along Z axis. 
        
    Returns
    -------
    None
    '''
    logging.info('bvecfile: {}'.format(bvecfile))
    logging.info('outfile: {}'.format(outfile))
    bvecs = np.loadtxt(bvecfile)
    if flip_x:
        logging.info('Flipping bvecs along X axis')
        bvecs[0,:] *= -1 
    if flip_y:
        logging.info('Flipping bvecs along Y axis')
        bvecs[1,:] *= -1 
    if flip_z:
        logging.info('Flipping bvecs along Z axis')
        bvecs[2,:] *= -1 
    bvecs += 0 
    logging.info('Writing bvecs file {}'.format(outfile))
    np.savetxt(outfile, bvecs, fmt='%0.8f')
    logging.info('End of Protocol {}'.format(PROTOCOL_NAME))
    
if __name__ == '__main__': 
    main(sys.argv[1:])
