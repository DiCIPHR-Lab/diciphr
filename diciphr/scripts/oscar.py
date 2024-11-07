#! /usr/bin/env python
import os, sys, subprocess, argparse, traceback
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from diciphr.oscar import Oscar 

DESCRIPTION='''OSCAR - A script to create screenshots from Nifti files and statistical results.'''

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-o','--output',action='store',metavar='<outbase>',dest='output_filebase',
                    type=str, required=True,
                    help='The output filebase. Will create a dir if it does not exist.'
                    )
    p.add_argument('-i','--underlay',action='store',metavar='<nii>',dest='underlay',
                    type=str, required=True,
                    help='The underlay image in Nifti format.'
                    )
    p.add_argument('-j','--overlay',action='append',metavar='<nii>',dest='overlays',
                    type=str, required=False, default=[],
                    help='An overlay image in Nifti format. Can be used more than once'
                    )
    p.add_argument('-c','--colormap',action='append',metavar='<cmap>',dest='cmaps',
                    type=str, required=False, default=[],
                    help='Provide a pyplot colormap for the overlay. Can be used more than once'
                    )
    p.add_argument('-m','--clim', action='append',metavar='<float>',dest='clims',
                    type=str, required=False, default=[],
                    help='Defines the range of the colorbar (clim), e.g. 1 for correlation coefficient. Can be used more than once.'
                    )
    p.add_argument('-s','--slicetype',action='store',metavar='<s/c/a>', dest='slice_type',
                    type=str, required=False, default='axial',
                    help='The slice type. Not yet implemented. Default axial.'
                    )
    p.add_argument('-n','--slicenum',action='store',metavar='<slice>', dest='slice_number',
                    type=int, required=False, default=[], nargs="*",
                    help='If 4D input, the slice chosen as background. Default is middle slice. If 3D input, expects a list of slices of the same length as nrows x ncolumns in grid mode.'
                    )
    p.add_argument('-g', '--grid', action='store', metavar='<int>', dest='grid', 
                    type=int, required=False, default=[], nargs="*",
                    help='Grid parameters, a space separated list of 4 values: nrows ncolumns center spacing. If 2 numbers are provided, -n option must be used to provide nrows x ncolumns values'
                    )                
    p.add_argument('--grid-view', action='store_true', dest='grid_view', 
                    required=False, default=False, 
                    help='Also save 3 views of the slices of your grid.'
                    )
    p.add_argument('-r','--framerate', action='store', metavar='<int>', dest='framerate',
                    required=False, default=5,
                    help='Framerate for movie.'
                    ) 
    p.add_argument('-t', '--format', action='store', metavar='<fmt>', dest='file_format', 
                    required=False, default='png',
                    help='Screenshot file type. .gif or .avi will produce screenshots and a gif/movie. Default is png.'
                    )
    p.add_argument('--pmask', action='store', metavar=['nii','alpha'], dest='pmask', 
                    type=str, required=False, default=['','0.05'], nargs=2, 
                    help='A Nifti image of p-values, will threshold overlay images at alpha value.'
                    )
    p.add_argument('--abs', action='store', metavar='threshold', dest='abs', 
                    type=float, required=False, default=0, 
                    help='Overlay voxels with absolute value less than this threshold will be set to 0.'
                    )
    p.add_argument('--title', action='store', metavar='<str>', dest='title', 
                    required=False, default='', 
                    help='A constant title for all screenshots taken.'
                    )
    p.add_argument('--figsize', action='store', dest='figsize', metavar='<float>', 
                    required=False, type=float, default=[6.0,4.0], nargs=2, 
                    help='The figure size, as values separated by spaces. Default is 6 4'
                    )
    p.add_argument('--dpi', action='store', dest='dpi', metavar='<int>', 
                    required=False, type=int, default='300', 
                    help='DPI of the figure. Default is 300'
                    )
    p.add_argument('-v', '--vmax', action='store', dest='vmax', metavar='<float>', 
                    required=False, type=float, default=0.5, 
                    help='Used to scale intensity of the underlay. Higher = darker. Default is 0.5'
                    )
    p.add_argument('-A', '--ulay-alpha', action='store', dest='ulay_alpha', metavar='<float>', 
                    required=False, type=float, default=1.0, 
                    help='Alpha for the underlay. Default is 1.0'
                    )
    p.add_argument('-a', '--olay-alpha', action='append', dest='olay_alphas', metavar='<float>', 
                    required=False, type=float, default=[], 
                    help='Alpha for the overlay(s). Default is 1.0'
                    )
    p.add_argument('--no-colorbar', action='store_true', dest='no_colorbar', 
                    required=False, default=False, 
                    help='Do not display colorbar for overlays.'
                    )
    p.add_argument('--bgcolor', action='store', dest='bgcolor', 
                    required=False, default='k', 
                    help='Background color, default is black.'
                    )
    p.add_argument('-x','--cleanup', action='store_true',dest='cleanup',
                    required=False, default=False,
                    help='Delete .pngs once movies have been made.')
    return p
    
   
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_filebase = os.path.realpath(args.output_filebase)
    output_dir = os.path.dirname(output_filebase)
    slice_type = args.slice_type
    slice_number = args.slice_number
    cleanup = args.cleanup
    grid = args.grid
    
    kwargs = {}
    kwargs['cmaps'] = args.cmaps 
    kwargs['file_format'] = args.file_format 
    kwargs['clims'] = args.clims 
    kwargs['vmax'] = args.vmax 
    kwargs['ulay_alpha'] = args.ulay_alpha 
    kwargs['olay_alphas'] = args.olay_alphas 
    kwargs['bgcolor'] = args.bgcolor 
    kwargs['figsize'] = args.figsize
    kwargs['dpi'] = args.dpi 
    kwargs['show_colorbar'] = not args.no_colorbar
    pmask_fn, palpha = args.pmask
    kwargs['palpha'] = float(palpha)
    kwargs['abs_thresh'] = args.abs
    if pmask_fn:
        kwargs['pmask'] = nib.load(pmask_fn)
    else:
        kwargs['pmask'] = None
    #Begin 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("underlay: {}".format(args.underlay))
    underlay_img = nib.load(args.underlay)
    if len(args.overlays) > 0: 
        print("overlays: {}".format(args.overlays))
    overlay_imgs = [ nib.load(o) for o in args.overlays ] 
    print("output_filebase: {}".format(output_filebase))
    print("slice_type: {}".format(slice_type))
    
    myOscar = Oscar(underlay_img, overlay_imgs=overlay_imgs, **kwargs)
    center = None
    
    # 4D input 
    if len(underlay_img.shape) > 3 and underlay_img.shape[-1] > 1:
        print("4D input detected")
        if args.file_format in ['gif','avi']:
            print("Creating frames for {} file".format(args.file_format))
            screenshots = myOscar.screenshots_4d(slice_type, slice_number[0], output_filebase)
            if args.file_format == 'gif':
                output_fn = myOscar.images_to_gif(screenshots, output_filebase+'.gif')
                print("Output gif: {}".format(output_fn))
            if args.file_format == 'avi':
                output_fn = myOscar.images_to_avi(output_filebase+'_%06d.png', output_filebase+'.avi', 
                    framerate=args.framerate)
                print("Output movie: {}".format(output_fn))
            if cleanup:
                for sc in screenshots: 
                    os.remove(sc)
        else:
            print("Create a grid for 4D Nifti data")
            myOscar.grid_4d(slice_type, slice_number[0], output_filebase, title=args.title)
    else:
        print("3D input detected")
        if grid:
            print("Create a grid of slices")
            ulay_center=None
            # if len(underlay_img.shape) > 3:
                # raise ValueError('Cannot produce a grid on a 4D underlay')
            if len(grid) == 4:
                nrows, ncols, center, spacing = grid
                slice_list = []
            else:
                if len(grid) == 3:
                    ulay_center = grid[2]
                if len(grid) <= 3:
                    nrows, ncols = grid[:2]
                    slice_list = slice_number
                    spacing=1
                    # if len(slice_list) != nrows * ncols:
                        # raise ValueError('Length of slice_number must match nrows * ncols')
                else:
                    raise ValueError('Improper number of grid arguments')
            if slice_type.lower() in ['sagittal','sag','s']:
                if center:
                    center = underlay_img.shape[0] - center
                if slice_list:
                    slice_list = sorted([underlay_img.shape[0] - s  for s in slice_list])
            
            myOscar.slice_grid(slice_type, output_filebase=output_filebase, slices=slice_list,
                        nrows=nrows, ncols=ncols, center=center, spacing=spacing, title=args.title)    
            if args.grid_view:
                myOscar.slice_grid_view(slice_type, output_filebase=output_filebase, slices=slice_list,
                        nrows=nrows, ncols=ncols, center=center, spacing=spacing, ulay_center=ulay_center)
        else:
            print("Create frames of slices from 3D Nifti data")
            screenshots = myOscar.screenshots_3d(slice_type, output_filebase)
            if args.file_format == 'gif':
                output_fn = myOscar.images_to_gif(screenshots, output_filebase+'.gif')
                print("Output gif: {}".format(output_fn))
            if args.file_format == 'avi':
                output_fn = myOscar.images_to_avi(output_filebase+'_%06d.png', output_filebase+'.avi', framerate=args.framerate)
                print("Output movie: {}".format(output_fn))
            if cleanup:
                for sc in screenshots: 
                    os.remove(sc)
            
if __name__ == '__main__':
    main(sys.argv[1:])
