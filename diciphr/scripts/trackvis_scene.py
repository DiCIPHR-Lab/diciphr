#! /usr/bin/env python

import os, sys, shutil, logging
from dipy.io.streamline import load_trk
from diciphr.utils import check_inputs, make_dir, protocol_logging, DiciphrArgumentParser
from diciphr.nifti_utils import strip_nifti_ext
from diciphr.tractography.track_utils import track_info, merge_tracks, downsample_tracks_fdc, create_scene
from collections import OrderedDict

DESCRIPTION = '''
    Merge multiple TRK files and create a Trackvis scene file with Nifti underlay and ROI images
'''

PROTOCOL_NAME='ROI_Stats'

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-t', '--trk', action='store', metavar='<path>', dest='trkfiles',
                    type=str, required=True, nargs="*",
                    help='Tractography file(s) in .trk format to be merged.'
                    )
    p.add_argument('-o', '--outbase', action='store', metavar='<path>', dest='outbase',
                    type=str, required=True, 
                    help='Basename of the merged trk file and scene file to be created. Output directory will be created if it does not exist.'
                    )
    p.add_argument('-d', '--downsample', action='store', metavar='<int>', dest='downsample_percent', 
                   type=int, default=100, 
                   help='Downsample percentage with fiber-density-coreset method. Default=100 (off)'
                   )
    p.add_argument('-m', '--minimum-fibers', action='store', metavar='<int>', dest='minimum_fibers', 
                   type=int, default=0, 
                   help='Do not downsample .trk files that are less than this number of fibers. Default=0 (off)'
                   )
    p.add_argument('-n', '--trknames', action='store', metavar='<str>', dest='track_names',
                    type=str, default=[], nargs="*",
                    help='Names of trk files to be rendered in the scene. Will default to the basename of the file with .trk extension removed.'
                    )
    p.add_argument('-c', '--trkcolors', action='store', metavar='<csv>', dest='track_colors',
                    type=str, default=None, 
                    help='A comma-separated values (csv) file with 4 columns: Track name, Red (0-255), Green (0-255), Blue (0-255)'
                    )
    p.add_argument('-i', '--underlays', action='store', metavar='<csv>', dest='underlays',
                    type=str, default=[], nargs="*",
                    help='Nifti files to add to the scene as underlay images'
                    )
    p.add_argument('-r', '--rois', action='store', metavar='<csv>', dest='rois',
                    type=str, default=[], nargs="*",
                    help='ROI Nifti files to add to the scene'
                    )
    
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    outdir = os.path.dirname(os.path.realpath(args.outbase))
    make_dir(outdir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        run_trackvis_scene(args)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise

default_colors_dict = OrderedDict()
default_colors_dict['AF'] = (0,255,0)       # AF-L green 
default_colors_dict['CC'] = (255,0,0)       # CC red 
default_colors_dict['CG'] = (255,255,0)     # cingulum yellow 
default_colors_dict['CR'] = (100, 149, 237) # Cornflower Blue
default_colors_dict['CST'] = (0,0,255)      # CST blue 
default_colors_dict['FX'] = (0,200,255)     # Fornix cyan
default_colors_dict['ICP'] = (0, 128, 128)  # Teal
default_colors_dict['ILF'] = (255, 26, 141) # ILF-L purplish red 
default_colors_dict['IFOF'] = (255,165,0)   # IFOF-L orange
default_colors_dict['MCP'] = (255, 215, 0)  # Gold
default_colors_dict['OR'] = (180, 0, 0)     # OR-L maroon 
default_colors_dict['SLF'] = (255,102,204)  # SLF pink
default_colors_dict['SLF1'] = (142,220,205) # SLF1-R blue-green
default_colors_dict['SLF2'] = (160,0,160)   # SLF2-L purple
default_colors_dict['SLF3'] = (255,102,204) # SLF3-L pink
default_colors_dict['UF'] = (128, 0, 255)   # Violet

class ColorCycle:
    def __init__(self, colors):
        self.i = 0 
        self.colors = colors
    def get_color(self):
        c = self.colors[self.i]
        self.i += 1 
        self.i = self.i % len(self.colors)
        return c 
unknown_colors = ColorCycle([(20, 120, 10), (190, 40, 90), (0, 255, 127), (70, 130, 180), (200, 160, 20)])
roi_colors = ColorCycle([(0,255,255),(255,255,0),(255,0,255),(255,0,0),(0,255,0),(0,0,255)])

def track_name_to_color(name, colors_dict):
    color = None
    name = name.replace('_','')
    for k in colors_dict.keys():
        if k.upper() in name.upper():
            color = colors_dict[k]
    if color is None:
        color = unknown_colors.get_color()
    return color

def run_trackvis_scene(args):
    trkfiles = args.trkfiles
    outbase = os.path.realpath(args.outbase)
    downsample_percent = args.downsample_percent
    minimum_fibers = args.minimum_fibers
    track_names = args.track_names
    track_colors = args.track_colors
    underlays = args.underlays
    rois = args.rois
    
    outdir = os.path.dirname(outbase)
    scenefile = outbase+'.scene'
    merged_trk_file = outbase+'.trk'
    
    logging.info("Input trk files:")
    check_inputs(*trkfiles)
    
    if track_colors:
        track_colors_dict = {}
        with open(track_colors, 'r') as fid:
            for line in fid.readlines():
                k, r, g, b = line.strip().split(',')
                track_colors_dict[k] = (int(r), int(g), int(b))
    else:
        track_colors_dict = default_colors_dict
    if not track_names:
        track_names = [os.path.basename(fn)[:-4] for fn in trkfiles]
    track_colors = [track_name_to_color(name, track_colors_dict) for name in track_names]
    
    for roi in rois:
        logging.info("Copying regions of interest to output directory")
        shutil.copyfile(roi, os.path.join(outdir, os.path.basename(roi)))
    for underlay in underlays:
        logging.info("Copying underlay images to output directory")
        shutil.copyfile(underlay, os.path.join(outdir, os.path.basename(underlay)))
    
    sfts = []
    for trkfile in trkfiles:
        sft = load_trk(trkfile, reference='same')
        nstreamlines = track_info(sft)['number streamlines']
        if nstreamlines > minimum_fibers and downsample_percent < 100:
            sft = downsample_tracks_fdc(sft, downsample_percent=downsample_percent, num_iters=15)
        sfts.append(sft)
    if len(sfts) > 1:
        logging.info("Merge Tracks")
    # merge_tracks will also add DataSetID descriptor 
    merge_tracks(sfts, merged_trk_file)
    os.chdir(outdir)
    
    underlays = [os.path.basename(fn) for fn in underlays]
    rois = [os.path.basename(fn) for fn in rois]
    
    create_scene(os.path.basename(merged_trk_file), 
                 os.path.basename(scenefile), 
                 underlay_images=underlays, 
                 roi_images=rois, 
                 roi_names=[strip_nifti_ext(fn) for fn in rois], 
                 roi_colors=[roi_colors.get_color() for r in rois], 
                 roi_opacities=[1 for r in rois],
                 track_group_names=track_names, 
                 track_colors=track_colors, 
                 solid_colors=True, 
                 render="Line")
    
if __name__ == '__main__':
    main(sys.argv[1:])
