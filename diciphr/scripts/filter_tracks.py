#! /usr/bin/env python

import os, sys, shutil, logging
from dipy.io.streamline import load_tractogram
from diciphr.utils import check_inputs, protocol_logging, TempDirManager, DiciphrArgumentParser
from diciphr.tractography.track_utils import ( filter_tracks_include, 
                            filter_tracks_exclude, track_density_image )
from diciphr.nifti_utils import read_nifti, resample_image

DESCRIPTION = '''
    Filter trk files 
'''

PROTOCOL_NAME='filter_tracks'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-f', action='store', metavar='trk', dest='input_trackfile',
                    type=str, required=True,
                    help='Path of the trackvis .trk file'
                    )
    p.add_argument('-o', action='store', metavar='output', dest='output_trackfile',
                    type=str, required=False,
                    help='Output track file'
                    )
    p.add_argument('-i', action='store', metavar='nii', dest='include_masks',
                    type=str, required=False, nargs="*", default=[], 
                    help='Include mask(s)'
                    )
    p.add_argument('-x', action='store', metavar='nii', dest='exclude_masks',
                    type=str, required=False, nargs="*", default=[], 
                    help='Exclude mask(s)'
                    )
    p.add_argument('-t','--tdi', action='store', metavar='tdi', dest='tdi_filename',
                    type=str, required=False, default=None, 
                    help='Output filename of a track density image (TDI). AFTER include/exclude'
                    )
    p.add_argument('-r','--ref', action='store', metavar='ref', dest='reference_nifti',
                    type=str, required=False, default=None, 
                    help='Reference Nifti, determines the grid and affine of the TDI image.'
                    )
    p.add_argument('-v','--voxelsize', action='store', metavar='float', dest='voxelsize', 
                    type=float, required=False, default=0.0,
                    help='Provide a desired voxel size of the TDI image. Affine and image size will be ascertained from the reference nifti.'
                    )
    return p
    
def main(argv):   
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.input_trackfile)
        if args.include_masks:
            check_inputs(*args.include_masks, nifti=True)
        if args.exclude_masks: 
            check_inputs(*args.exclude_masks, nifti=True)
        if args.output_trackfile:
            run_filter_tracks(args.input_trackfile, args.output_trackfile, args.include_masks, args.exclude_masks)
        else:
            args.output_trackfile = args.input_trackfile
        if args.tdi_filename:
            run_track_density_image(args.output_trackfile, args.tdi_filename, args.reference_nifti, args.voxelsize)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def run_track_density_image(input_trackfile, output_tdifile, reference_nifti, voxel=0.0):
    logging.info('TDI Image')
    logging.info('Load reference image')
    ref_im = read_nifti(reference_nifti)
    if voxel > 0.0: 
        logging.info(f'Resample reference image to {voxel} isotropic')
        ref_im = resample_image(ref_im, (voxel, voxel, voxel), interp='nearest')
    logging.info('Calculate the TDI image')
    streamlines = load_tractogram(input_trackfile, 'same')
    tdi_im = track_density_image(streamlines.streamlines, ref_im)
    tdi_im.to_filename(output_tdifile)
    logging.info(f'Saved file {output_tdifile}')
   
def run_filter_tracks(input_trackfile, output_trackfile, include_masks, exclude_masks):
    logging.info(f'Filter tracks through {include_masks} excluding {exclude_masks}')
    with TempDirManager(prefix='filter_tracks') as manager:
        tmpdir = manager.path()
        working_trackfile = input_trackfile
        if include_masks:
            logging.info(f'Include masks: {include_masks}')
            trk_include = os.path.join(tmpdir, 'filter_include.trk')
            filter_tracks_include(working_trackfile, trk_include, include_masks)
            working_trackfile = trk_include
        if exclude_masks:
            logging.info(f'Exclude masks: {exclude_masks}')
            trk_exclude = os.path.join(tmpdir, 'filter_exclude.trk')
            filter_tracks_exclude(working_trackfile, trk_exclude, exclude_masks)
            working_trackfile = trk_exclude
        logging.info(f'Write tracks to {output_trackfile}')
        shutil.copyfile(working_trackfile, output_trackfile)

if __name__ == '__main__': 
    main(sys.argv[1:])
