#! /usr/bin/env python

import os, sys, logging 
from diciphr.utils import protocol_logging, DiciphrArgumentParser, DiciphrException
from diciphr.tractography.track_utils import track_density_image 
import nibabel
import numpy as np

DESCRIPTION = '''
    Convert between tractography file types.
'''

PROTOCOL_NAME='convert_tracks'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-i', action='store', metavar='input', dest='input_files',
                    type=str, required=True, nargs="*",
                    help='The input file(s)'
                    )
    p.add_argument('-o', action='store', metavar='output', dest='output_files',
                    type=str, required=True, nargs="*",
                    help='The output file(s)'
                    )
    p.add_argument('-a', action='store', metavar='ref', dest='ref_nifti',
                    type=str, required=False, default=None,
                    )
    return p

def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        convert_tracks(args.input_files, args.output_files, ref_nifti=args.ref_nifti)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

def convert_tracks(input_files, output_files, ref_nifti=None):
    if not len(input_files) == len(output_files):
        raise DiciphrException('Length of input_files must match length of output_files!')
    if ref_nifti is not None:
        ref_im = nibabel.load(ref_nifti)
    for input_file,output_file in zip(input_files, output_files):
        logging.info('Open {}'.format(input_file))
        if not nibabel.streamlines.is_supported(input_file):
            raise DiciphrException('Filetype not supported: {}'.format(input_file))
        ext = output_file.split('.')[-1]
        if not ext in ['trk','tck','nii','gz']:
            raise DiciphrException('Only trk, tck, nii, nii.gz files currently supported!')
        logging.info('Read streamlines')
        T_in = nibabel.streamlines.load(input_file, lazy_load=False)
        tractogram = T_in.tractogram
        if ext == 'trk':
            header = nibabel.streamlines.TrkFile.create_empty_header()
            if ref_nifti is not None:
                header['voxel_to_rasmm'] = ref_im.affine.copy() # voxel_to_rasmm
                header['voxel_order'] = "".join(nibabel.aff2axcodes(ref_im.affine)) # voxel_order
                header['voxel_sizes'] = ref_im.header.get_zooms()[:3] # voxel_sizes
                header['dimensions'] = ref_im.shape[:3] # dimensions
            logging.info('Save Trackvis format file to {}'.format(output_file))
            nibabel.streamlines.save(tractogram, output_file, header=header)
        elif ext == 'tck':  
            header = nibabel.streamlines.TckFile.create_empty_header()
            logging.info('Save Mrtrix format file to {}'.format(output_file))
            nibabel.streamlines.save(tractogram, output_file, header=header)
        elif ext == 'nii' or ext == 'gz':
            if ref_nifti is None:
                raise DiciphrException('Reference nifti is required to calculate track density image.')
            logging.info('Calculate track density image')
            tdi_im = track_density_image(T_in.streamlines, ref_im) 
            logging.info('Save NiFTI track density image file to {}'.format(output_file))
            tdi_im.to_filename(output_file)
        
if __name__ == '__main__':
    main(sys.argv[1:])
