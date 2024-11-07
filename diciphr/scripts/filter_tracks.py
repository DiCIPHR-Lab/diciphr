#! /usr/bin/env python

import os, sys, shutil, logging, traceback, argparse
import nibabel as nib
from nibabel.trackvis import read as read_trk
from dipy.io.streamline import load_tractogram
from ..utils import ( check_inputs, make_dir, make_temp_dir, protocol_logging, DiciphrException )
from ..tractography.track_utils import filter_tracks_include, filter_tracks_exclude, track_density_image
from ..nifti_utils import add_images, resample_image

DESCRIPTION = '''
    Filter trk files 
'''

PROTOCOL_NAME='filter_tracks'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-f', action='store', metavar='fiberfile', dest='input_trackfile',
                    type=str, required=True,
                    help='Path of the trackvis .trk file'
                    )
    p.add_argument('-o', action='store', metavar='output', dest='output_trackfile',
                    type=str, required=False,
                    help='Output track file'
                    )
    p.add_argument('-i', action='append', metavar='include_mask', dest='include_masks',
                    type=str, required=False, default=[], 
                    help='Include masks, if option is used more than once, will be an AND operator'
                    )
    p.add_argument('-x', action='append', metavar='exclude_mask', dest='exclude_masks',
                    type=str, required=False, default=[], 
                    help='Exclude masks, if option is used more than once, will be an OR operator'
                    )
    p.add_argument('-t','--tdi', action='store', metavar='tdi', dest='tdi_filename',
                    type=str, required=False, default=None, 
                    help='Output filename of a track density image (TDI). AFTER include/exclude'
                    )
    p.add_argument('-r','--ref', action='store', metavar='ref', dest='reference_nifti',
                    type=str, required=False, default=None, 
                    help='Reference Nifti, determines the grid and affine of the TDI image.'
                    )
    p.add_argument('-v','--voxelsize', action='store', metavar='voxelsize', dest='voxelsize', 
                    type=float, required=False, default=0.0,
                    help='Provide a desired voxel size of the TDI image. Affine and image size will be ascertained from the reference nifti.'
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
    # 2. Parse command line args    
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, args.logfile, debug=args.debug)
    try:
        # 5. Check necessary inputs exist
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
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e

def run_track_density_image(input_trackfile, output_tdifile, reference_nifti, voxel=0.0):
    logging.info('TDI Image')
    logging.info('Load reference image')
    ref_im = nib.load(reference_nifti)
    if voxel > 0.0: 
        logging.info('Resample reference image to {} isotropic'.format(voxel))
        ref_im = resample_image(ref_im, (voxel, voxel, voxel), interp='nearest')
    logging.info('Calculate the TDI image')
    streamlines = load_tractogram(input_trackfile, 'same')
    tdi_im = track_density_image(streamlines.streamlines, ref_im)
    tdi_im.to_filename(output_tdifile)
    logging.info('Saved file {}'.format(output_tdifile))
    
def run_filter_tracks(input_trackfile, output_trackfile, include_masks, exclude_masks):
    logging.info('Filter tracks through {} excluding {}'.format(include_masks, exclude_masks))
    tmpdir=make_temp_dir(prefix='filter_tracks')
    try:
        exclude_mask = None
        if exclude_masks:
            logging.debug('Add all exclusion masks into one nifti')
            exclude_masks = [ nib.load(_) for _ in exclude_masks ]
            exclude_mask = add_images(*exclude_masks, binarize=True)
            exclude_mask.to_filename(os.path.join(tmpdir, 'exclude.nii'))
            exclude_mask = os.path.join(tmpdir, 'exclude.nii')
        counter=1
        input_trk_file = input_trackfile
        final_trk_file = input_trackfile
        for include_mask in include_masks:
            logging.info('Inclusion mask {0}'.format(counter))
            output_trk_file = os.path.join(tmpdir, 'include_c{0}.trk'.format(counter))
            logging.debug('diciphr.tractography.filter_tracks_include write to {}'.format(output_trk_file))
            filter_tracks_include(input_trk_file, include_mask, output_trk_file)
            counter=counter+1
            input_trk_file = output_trk_file
            final_trk_file = output_trk_file
        if exclude_mask:
            logging.info('Exclusion mask')
            output_trk_file = os.path.join(tmpdir, 'exclude.trk')
            final_trk_file = output_trk_file
            logging.debug('diciphr.tractography.filter_tracks_exclude write to {}'.format(output_trk_file))
            filter_tracks_exclude(input_trk_file, exclude_mask, output_trk_file)
        logging.info('Write tracks to {}'.format(output_trackfile))
        shutil.copyfile(final_trk_file, output_trackfile)
    except Exception as e:
        logging.error(str(e))
        raise e
    finally:
        shutil.rmtree(tmpdir)    

if __name__ == '__main__': 
    main(sys.argv[1:])
