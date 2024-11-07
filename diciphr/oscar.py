# -*- coding: utf-8 -*-
"""
Created on Fri Aug 2 2019

@author: parkerwi
"""

import os, sys, shutil, logging, traceback
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
plt.ioff()
from .nifti_utils import reorient_nifti, nifti_image, multiply_images 
from .utils import DiciphrException, ExecCommand
from math import ceil 

def absolute_threshold_image(nifti_img, threshold=0.0):
    data = nifti_img.get_data()
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
        self.vmax = kwargs.get('vmax',1.0)*np.max(underlay_img.get_data())
        self.show_colorbar = kwargs.get('show_colorbar',True)
        file_format = kwargs.get('file_format','png')
        self.figsize = kwargs.get('figsize', [6.,4.])
        self.dpi = kwargs.get('dpi', 350)
        pmask = kwargs.get('pmask', None)
        palpha = kwargs.get('palpha', 0.05)
        self.abs_thresh = kwargs.get('abs_thresh', 0.0)
        if pmask is not None:
            _pmask = nifti_image((pmask.get_data() < float(palpha))*1, pmask.affine)
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

                C = self.auto_contrast(o.get_data())
            else:
                try:
                    if ',' in str(c):
                        C = [float(i) for i in c.split(',')][0:2]
                    else:
                        if o.get_data().min() >= 0:
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
        v99 = np.percentile(data, 99)
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
        data = nifti_img.get_data()
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
            'vmax':self.vmax/np.max(self.underlay_img.get_data()),
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
            raise DiciphrException('Grid out of range!')
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

    def screenshots_4d(self, slice_type, slice_number, output_filebase, 
                axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15):
        underlay = self.underlay_img.get_data()
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
            p = self.plot_slice(ax, i, data)
            if title:
                fig.suptitle(title)
            plt.savefig(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format), dpi=self.dpi, facecolor=fig.get_facecolor())
            filenames.append(output_filebase+'_{0:06d}.{1}'.format(i, self.file_format))
            plt.close()
        return filenames
        
    def grid_4d(self, slice_type, slice_number, output_filebase, 
            axes_pad=0.0, cbar_location="right", cbar_size="5%", cbar_pad=0.15, title=None):
        underlay = self.underlay_img.get_data()
        nt = underlay.shape[-1]
        if nt < 10:
            ncols = nt
        else:
            ncols = 10 
        nrows = int(ceil(float(nt) / ncols))
        filenames = []
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
            p = self.plot_slice(ax, slice_number, data)
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