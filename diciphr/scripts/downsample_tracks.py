#! /usr/bin/env python 

import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging, 
                DiciphrArgumentParser, DiciphrException )
from diciphr.nifti_utils import read_nifti
from diciphr.tractography.track_utils import downsample_tracks
from dipy.io.streamline import load_trk, save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from nibabel.trackvis import TrackvisFile 

DESCRIPTION = '''
    Downsample a .trk file by a given percentage. 
'''

PROTOCOL_NAME='downsample_tracks'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', action='store', metavar='input', dest='input_trk',
                    type=str, required=True,
                    help='Input .trk filename.'
                    )
    p.add_argument('-o', action='store', metavar='output', dest='output_trk',
                    type=str, required=True,
                    help='Output .trk filename.'
                    )
    p.add_argument('-r', action='store', metavar='nifti', dest='ref_nifti',
                    type=str, required=True,
                    help='Reference Nifti file.'
                    )
    p.add_argument('-p', action='store', metavar='percent', dest='downsample_percent',
                    type=int, required=True, 
                    help='Target percentage of tracks to keep. Integer between 1 and 99.'
                    )
    p.add_argument('-n', action='store', metavar='numiters', dest='num_iters', 
                    type=int, required=False, default=15,
                    help='Number of iterations to run. Default is 15.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        input_trk = check_inputs(args.input_trk)
        ref_nifti = check_inputs(args.ref_nifti)
        output_trk = os.path.realpath(args.output_trk)
        output_dir = os.path.dirname(os.path.realpath(args.output_trk))
        downsample_percent = args.downsample_percent
        num_iters = args.num_iters
        make_dir(output_dir, recursive=True, pass_if_exists=True)
        protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
        logging.info('Reading image file {}'.format(ref_nifti))
        ref_im = read_nifti(ref_nifti)
        logging.info('Reading track file {}'.format(input_trk))
        tractogram = load_trk(input_trk, ref_im)
        streams = tractogram.get_streamlines_copy()
        streams_downsampled = downsample_tracks(streams, ref_im, 
                        downsample_percent=downsample_percent,
                        num_iters=num_iters)
        logging.info('Write to output {}'.format(output_trk))
        tractogram = StatefulTractogram(streams_downsampled, ref_im, Space.RASMM)
        save_trk(tractogram, output_trk)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
if __name__ == '__main__':
    main(sys.argv[1:])
