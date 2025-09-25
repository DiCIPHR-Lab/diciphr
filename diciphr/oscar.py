# -*- coding: utf-8 -*-
"""
Created on Fri Aug 2 2019

@author: parkerwi
"""

import os, logging
import argparse 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.ioff()
from diciphr.nifti_utils import read_nifti, reorient_nifti, nifti_image, multiply_images 
from diciphr.utils import DiciphrException, ExecCommand
from math import ceil 

DESCRIPTION = '''OSCAR - Utility to Overlay Statistical Content on Anatomical Reference '''

def absolute_threshold_image(nifti_img, threshold=0.0):
    data = nifti_img.get_fdata()
    data[np.abs(data)<threshold] = 0 
    return nifti_image(data, nifti_img.affine)

class Oscar(object):
    def __init__(self, underlay_img, overlay_imgs=[], **kwargs):
        self.underlay_img = underlay_img
        self.affine = underlay_img.affine
        self.overlay_imgs = overlay_imgs
        self.N = len(self.overlay_imgs)
        self.bgcolor = kwargs.get('bgcolor','w')
        self.clim_args = kwargs.get('clims',['1.0'])
        self.cmaps = kwargs.get('cmaps',['jet'])
        self.olay_alphas = kwargs.get('olay_alphas', [1.0])
        self.ulay_alpha = kwargs.get('ulay_alpha',0.7)
        self.vmax = kwargs.get('vmax',1.0)*np.max(underlay_img.get_fdata())
        self.show_colorbar = kwargs.get('show_colorbar',True)
        file_format = kwargs.get('file_format','png')
        self.figsize = kwargs.get('figsize', [6.,4.])
        self.dpi = kwargs.get('dpi', 350)
        pmask = kwargs.get('pmask', None)
        palpha = kwargs.get('palpha', 0.05)
        self.abs_thresh = kwargs.get('abs_thresh', 0.0)
        if pmask is not None:
            _pmask = nifti_image((pmask.get_fdata() < float(palpha))*1, pmask.affine)
            self.overlay_imgs = [ multiply_images(olay_img, _pmask) for olay_img in self.overlay_imgs ]
        if self.abs_thresh > 0:
            self.overlay_imgs = [ absolute_threshold_image(olay_img, self.abs_thresh) for olay_img in self.overlay_imgs ]
        if file_format in ['gif','avi']:
            self.file_format = 'png'
            self.movie_format = file_format
        else:
            self.file_format = file_format
            self.movie_format = None
        # ensure lengths of relevant lists are same, set up clims etc. 
        if self.N > 0:
            self._configure_olays()
        else:
            # no overlays means no colorbar 
            self.show_colorbar = False 
        
    def _configure_olays(self):
        if len(self.cmaps) < self.N:
            self.cmaps += ['jet' for a in range(self.N - len(self.cmaps))]
        elif len(self.cmaps) > self.N:
            self.cmaps = self.cmaps[:self.N]
        if len(self.clim_args) < self.N:
            self.clim_args += ['auto' for a in range(self.N - len(self.clim_args))]
        if len(self.clim_args) > self.N:
            self.clim_args = self.clim_args[:self.N]
        if len(self.olay_alphas) < self.N:
            self.olay_alphas += [1.0 for a in range(self.N - len(self.olay_alphas))]
        if len(self.olay_alphas) > self.N:
            self.olay_alphas = self.olay_alphas[:self.N]
        # configure clims 
        self.clims = [] 
        for c, o in zip(self.clim_args, self.overlay_imgs):
            if str(c).lower() == 'auto':

                C = self.auto_contrast(o.get_fdata())
            else:
                try:
                    if ',' in str(c):
                        C = [float(i) for i in c.split(',')][0:2]
                    else:
                        if o.get_fdata().min() >= 0:
                            C = [0.0, float(c)]
                        else:
                            C = [-1*float(c), float(c)]
                except:
                    raise DiciphrException('Could not understand clim argument {}'.format(c))
            self.clims.append(C)
            
    @staticmethod
    def auto_contrast(data):
        vmin = float(np.min(data))
        vmax = float(np.max(data))
        # to do - winsorize
        if 0.9 < vmax <= 1.0:
            vmax = 1.0 
        else:
            vmax = 0.95 * vmax 
        if vmin >= 0:
            vmin = 0
        elif -1.0 <= vmin <= -0.9:
            vmin = -1.0 
        else:
            vmin = 0.95 * vmin 
        return [vmin, vmax]
        
    @staticmethod
    def get_slice_data(nifti_img, slice_type, bgvalue=0):
        ''' Creates an axis-reordered version of your nifti_img data,
        given a slice_type ('sagittal', 'coronal', 'axial'), 
        and ready to be plotted using matplotlib.pyplot.imshow function.'''
        nifti_img = reorient_nifti(nifti_img, 'RAS')
        data = nifti_img.get_fdata()
        data = np.ma.masked_where(data == bgvalue, data)
        if slice_type.lower() in ['axial','ax','a','z']:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
        elif slice_type.lower() in ['coronal','cor','c','y']:
            data = np.flip(data, axis=0)
            data = np.rollaxis(data, 2, 1)
            data = np.flip(data, axis=1)
            data = np.flip(data, axis=2) 
        elif slice_type.lower() in ['sagittal','sag','s','x']:
            data = np.moveaxis(data, 0, 2)
            data = np.flip(data, axis=1)
            data = np.flip(data, axis=0)
        else:
            raise DiciphrException('Slice type keyword not recognized: {}'.format(slice_type))
        return data 
    
    @staticmethod
    def calc_grid_slices(nrows, ncols, center, spacing):
        n_grid = nrows * ncols 
        min_slice = center - (int(n_grid/2)) * spacing
        slices = [ min_slice + i*spacing  for i in range(n_grid)]
        logging.info("Grid slices: {}".format(','.join(map(str,slices))))
        return slices 
        
    @staticmethod 
    def images_to_gif(filenames, output_filename):
        import imageio
        images=[]
        for fn in filenames:
            images.append(imageio.imread(fn))
        imageio.mimsave(output_filename, images)
        return output_filename
        
    @staticmethod 
    def images_to_avi(search_string, output_filename, framerate=5):
        ffmpeg_cmd=['ffmpeg', '-f', 'image2', '-framerate', str(int(framerate)), '-i', search_string, output_filename]
        ExecCommand(ffmpeg_cmd).run()
        return output_filename
    
    def plot_on_axis(self, axis, slice_type, slice_number, cmaps=None, clims=None, olay_alphas=None):
        underlay_data = self.get_slice_data(self.underlay_img, slice_type)
        overlay_datas = [self.get_slice_data(o_img, slice_type) for o_img in self.overlay_imgs]
        self.plot_slice(axis, slice_number, underlay_data, overlay_datas, cmaps, clims, olay_alphas)
        
    def plot_slice(self, ax, slice, underlay_data, overlay_datas=[], cmaps=None, clims=None, olay_alphas=None):
        ax.set_axis_off()
        p = ax.imshow(underlay_data[:,:,slice].T, cmap='gray', interpolation='none', 
                    origin='upper', vmax=self.vmax, vmin=underlay_data.min(), 
                    zorder=1, alpha=self.ulay_alpha)
        if len(overlay_datas) > 0:
            if cmaps is None:
                cmaps = self.cmaps 
            if clims is None:
                clims = self.clims 
            if olay_alphas is None:
                olay_alphas = self.olay_alphas 
            for i, (overlay_data, cmap, clim, olay_alpha) in enumerate(
                    zip(overlay_datas, cmaps, clims, olay_alphas)
                ):
                p = ax.imshow(overlay_data[:,:,slice].T, cmap=cmap, interpolation='none', clim=clim, zorder=i+2, alpha=olay_alpha)
        return p 

    def slice_grid_view(self, slice_type, output_filebase='', nrows=1, ncols=1, center=None, spacing=1, 
                slices=[], ulay_slice_type=None, ulay_center=None):
        figsize = [ a/2 for a in self.figsize ] 
        kwargs = {
            'clims':['0.0,1.0'],
            'cmaps':['hsv'],
            'olay_alphas':[1.0],
            'show_colorbar':False, 
            'bgcolor':self.bgcolor, 
            'dpi':self.dpi, 
            'figsize':figsize,
            'file_format':self.file_format,
            'vmax':self.vmax/np.max(self.underlay_img.get_fdata()),
         }
        if len(slices) == 0:
            slices = self.calc_grid_slices(nrows, ncols, center, spacing)
        if slice_type.lower() in ['sagittal','sag','s']:
            overlay_data = np.zeros(self.underlay_img.shape)
            overlay_data[np.array(slices),:,:] = 1
            overlay_img = nifti_image(overlay_data, self.affine)
            if ulay_slice_type is None:
                ulay_slice_type='axial'
        elif slice_type.lower() in ['coronal','cor','c']:
            overlay_data = np.zeros(self.underlay_img.shape)
            overlay_data[:,np.array(slices),:] = 1
            overlay_img = nifti_image(overlay_data, self.affine)
            if ulay_slice_type is None:
                ulay_slice_type='sagittal'
        elif slice_type.lower() in ['axial','ax','a']:
            overlay_data = np.zeros(self.underlay_img.shape)
            overlay_data[:,:,np.array(slices)] = 1
            overlay_img = nifti_image(overlay_data, self.affine)
            if ulay_slice_type is None:
                ulay_slice_type='sagittal'
        subOscar = Oscar(self.underlay_img, overlay_imgs=[overlay_img], **kwargs)
        fig = subOscar.slice_grid(ulay_slice_type, center=ulay_center)
        if output_filebase:
            plt.savefig('{0}_grid_{1}.{2}'.format(output_filebase, slice_type, self.file_format), dpi=self.dpi, 
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
        else:
            return fig 
        
    def slice_grid(self, slice_type, output_filebase='', nrows=1, ncols=1, center=None, spacing=1, 
                slices=[], axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15,
                title=''):
        underlay_data = self.get_slice_data(self.underlay_img, slice_type)
        overlay_datas = [ self.get_slice_data(o, slice_type) for o in self.overlay_imgs ] 
        nz = underlay_data.shape[-1]
        if center is None:
            center = int(nz/2)
        if len(slices) == 0:
            slices = self.calc_grid_slices(nrows, ncols, center, spacing)
        if slices[0] < 0 or slices[-1] > nz: 
            raise DiciphrException(f'Grid out of range! nslices={nz} grid={slices}')
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        grid = ImageGrid(fig, 111,
                nrows_ncols=(nrows, ncols),
                axes_pad=axes_pad,
                share_all=True,
                cbar_location=cbar_location,
                cbar_mode="single" if self.show_colorbar else None,
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
        )
        for slice, ax in zip(slices, grid):
            p = self.plot_slice(ax, slice, underlay_data, overlay_datas=overlay_datas)
            ax.set_facecolor(self.bgcolor)
        if self.show_colorbar:
            cbar = ax.cax.colorbar(p)  #last one 
            ax.cax.toggle_label(True)
            cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels') 
            if self.bgcolor.startswith('w'):            
                plt.setp(cbytick_obj, color='k')
            else:
                plt.setp(cbytick_obj, color='w')
        fig.set_facecolor(self.bgcolor)
        if title:
            title_obj = fig.suptitle(title)
            if self.bgcolor.startswith('w'):       
                plt.setp(title_obj, color='k')     
            else:
                plt.setp(title_obj, color='w')     
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        if output_filebase:
            plt.savefig('{0}.{1}'.format(output_filebase, self.file_format), dpi=self.dpi, 
                facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            logging.info('Saved image file ' + '{0}.{1}'.format(output_filebase, self.file_format))
            plt.close()
        else:
            return fig 
    
    def screenshots_3d(self, slice_type, output_filebase, title='',
                axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15):
        underlay_data = self.get_slice_data(self.underlay_img, slice_type)
        overlay_datas = [ self.get_slice_data(o, slice_type) for o in self.overlay_imgs ] 
        
        filenames=[]
        nz = underlay_data.shape[-1]
        for i in range(nz):
            logging.info("Slice {} out of {}".format(i+1,nz))
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            fig.set_facecolor(self.bgcolor)
            grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 1),
                    axes_pad=axes_pad,
                    share_all=True,
                    cbar_location=cbar_location,
                    cbar_mode="single" if self.show_colorbar else None,
                    cbar_size=cbar_size,
                    cbar_pad=cbar_pad,
            )
            ax = grid[0]
            ax.set_facecolor(self.bgcolor)
            p = self.plot_slice(ax, i, underlay_data, overlay_datas=overlay_datas)
            if self.show_colorbar:
                cbar = ax.cax.colorbar(p)
                ax.cax.toggle_label(True)
                cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels') 
                if self.bgcolor == 'w':
                    plt.setp(cbytick_obj, color='k')
                else:
                    plt.setp(cbytick_obj, color='w')
            if title:
                fig.suptitle(title)
            plt.savefig(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format), dpi=self.dpi, facecolor=fig.get_facecolor())
            filenames.append(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format))
            plt.close()
        return filenames

    def screenshots_4d(self, slice_type, slice_number, output_filebase, title='',
                axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15):
        underlay = self.underlay_img.get_fdata()
        nt = underlay.shape[-1]
        filenames = []
        # turn the image into a list of 3d niftis, squeeze because tensors are x,y,z,1,6 
        for i in range(nt):
            logging.info("Slice {} out of {}".format(i+1,nt))
            img = nifti_image(np.squeeze(underlay[...,i]), self.affine)
            data = self.get_slice_data(img, slice_type)[:,:,slice_number]
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            fig.set_facecolor(self.bgcolor)
            grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, 1),
                    axes_pad=axes_pad,
                    share_all=True,
                    cbar_location=cbar_location,
                    cbar_mode="single" if self.show_colorbar else None,
                    cbar_size=cbar_size,
                    cbar_pad=cbar_pad,
            )
            ax = grid[0]
            ax.set_facecolor(self.bgcolor)
            self.plot_slice(ax, i, data)
            if title:
                fig.suptitle(title)
            plt.savefig(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format), dpi=self.dpi, facecolor=fig.get_facecolor())
            filenames.append(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format))
            plt.close()
        return filenames
        
    def grid_4d(self, slice_type, slice_number, output_filebase, 
            axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15, title=None):
        underlay = self.underlay_img.get_fdata()
        nt = underlay.shape[-1]
        if nt < 10:
            ncols = nt
        else:
            ncols = 10 
        nrows = int(ceil(float(nt) / ncols))
        # turn the image into a list of 3d niftis, squeeze because tensors are x,y,z,1,6 
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.set_facecolor(self.bgcolor)
        grid = ImageGrid(fig, 111,
                nrows_ncols=(nrows, ncols),
                axes_pad=axes_pad,
                share_all=True,
                cbar_location=cbar_location,
                cbar_mode="single" if self.show_colorbar else None,
                cbar_size=cbar_size,
                cbar_pad=cbar_pad,
        )
        if slice_number is None:
            img = nifti_image(np.squeeze(underlay[...,0]), self.affine)
            slice_number = int(self.get_slice_data(img, slice_type).shape[-1] / 2)
        for i in range(nrows*ncols):
            if i < nt:
                img = nifti_image(np.squeeze(underlay[...,i]), self.affine)
            else:
                # the grid is larger than the nifti. plot zeros for remainder 
                img = nifti_image(np.zeros(underlay.shape[:3]), self.affine)
            data = self.get_slice_data(img, slice_type)
            ax = grid[i]
            ax.set_facecolor(self.bgcolor)
            self.plot_slice(ax, slice_number, data)
        if title:
            title_obj = fig.suptitle(title, fontsize=16)
            if self.bgcolor.startswith('w'):       
                plt.setp(title_obj, color='k')     
            else:
                plt.setp(title_obj, color='w')    
        fig.set_facecolor(self.bgcolor)
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.savefig('{0}.{1}'.format(output_filebase, self.file_format), dpi=self.dpi, 
                facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()    
        return '{0}.{1}'.format(output_filebase, self.file_format)

### OSCAR COMMAND LINE SCRIPT 

def oscar_argparser():
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
    
def run_oscar_commandline(args):
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
        kwargs['pmask'] = read_nifti(pmask_fn)
    else:
        kwargs['pmask'] = None
    #Begin 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("underlay: {}".format(args.underlay))
    underlay_img = read_nifti(args.underlay)
    if len(args.overlays) > 0: 
        print("overlays: {}".format(args.overlays))
    overlay_imgs = [ read_nifti(o) for o in args.overlays ] 
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
