# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, sys, shutil, logging, traceback
from collections import OrderedDict
from .utils import ( make_dir, make_temp_dir, which, 
                ExecCommand, DiciphrException, logical_and, force_to_list )
from .nifti_utils import (read_nifti, reorient_nifti,
    read_dwi, reorient_dwi, strip_nifti_ext )
try:
    from pydicom import read_file as _read_dicom_file
except ImportError:
    from dicom import read_file as _read_dicom_file

#############################################
############  DICOM Utilities  ##############
#############################################
def read_dicom_file(filename):
    logging.debug('diciphr.dicoms.read_dicom_file')
    tmpdir = None
    try:
        dcm = _read_dicom_file(filename, force=True)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)
    return dcm

def get_dicom_series_attributes(dicom_files, keys=None, replace_spaces=False, ignore_attribute_error=True):
    logging.debug('diciphr.dicoms.get_dicom_series_attributes')
    def _all_are_same(input_list):
        return len(set(map(str,input_list))) <= 1 
    if len(dicom_files) == 0:
        raise DiciphrException('Input sequence of dicom files is empty.')
    if keys is None:
        keys = [
            'PatientID',
            'AcquisitionDate',
            'SeriesDescription',
            'SeriesNumber',
            'PixelSpacing',
            'SliceThickness',
            'RepetitionTime',
            'EchoTime',
            'MagneticFieldStrength',
            'Manufacturer',
            'ManufacturerModelName',
            'InstitutionName'
        ]
    alt_keys = {
        'AcquisitionDate':'StudyDate',
        'SeriesDescription':'ProtocolName'
    }
    dicom_attributes = OrderedDict( (_k, []) for _k in keys )
    # dicom_attributes = {}
    if ignore_attribute_error:
        default=''
    else:
        default='raise'
    for dicom_file in dicom_files: 
        try:
            dcm = read_dicom_file(dicom_file)
        except:
            logging.error('File not read by dicom module {}'.format(dicom_file))
            continue
        for key in keys:
            try:
                try:
                    _attribute = getattr(dcm,key)
                except AttributeError as e:
                    if key in alt_keys:
                        _attribute = getattr(dcm,alt_keys[key])
                    else:
                        raise e
                if replace_spaces:
                    _attribute = '_'.join(str(_attribute).split())
                dicom_attributes[key].append(_attribute)
            except AttributeError as e:
                if ignore_attribute_error:
                    logging.warning("Dicom {0} AttributeError {1}".format(dicom_file, repr(e)))
                    dicom_attributes[key].append('')
                else:
                    raise e
    return dicom_attributes

def sort_and_link_dicoms_by_series_attributes(dicom_files, output_dir):
    logging.debug('diciphr.dicoms.sort_and_link_dicoms_by_series_attributes')
    attribute_keys = [
        'SeriesNumber',
        'SeriesDescription',
        'AcquisitionDate'
    ]
    dicom_lookup = {}
    out_directories = set()
    dicom_fails = []
    for dicom_file in dicom_files: 
        try:
            dcm = read_dicom_file(dicom_file)
        except:
            logging.warning('File not read by dicom module {}'.format(dicom_file))
            dicom_fails.append(dicom_file)
            continue
        try:
            dicom_lookup[dicom_file] = get_dicom_series_attributes([dicom_file], keys=attribute_keys, replace_spaces=True, ignore_attribute_error=True)
            logging.debug(dicom_lookup[dicom_file])
            attributes_dir_name = '_'.join([
                    dicom_lookup[dicom_file]['SeriesNumber'][0].replace(os.sep,''),
                    dicom_lookup[dicom_file]['SeriesDescription'][0].replace(os.sep,''),
                    dicom_lookup[dicom_file]['AcquisitionDate'][0].replace(os.sep,'')
            ])
            out_dir = os.path.join(output_dir, attributes_dir_name)
            make_dir(out_dir, pass_if_exists=True)
            link_filename=os.path.join(out_dir, os.path.basename(dicom_file))
            if not os.path.exists(link_filename):
                os.symlink(os.path.realpath(dicom_file), link_filename)
            out_directories.add(out_dir)
        except Exception as e:
            logging.error('Failed to link dicom {} by series attributes'.format(dicom_file))
            logging.error(repr(e))
            dicom_fails.append(dicom_file)
            continue
    if dicom_fails:
        logging.warning('Dicoms that failed diciphr.nifti_utils.sort_and_link_dicoms_by_series_attributes: {}'.format(dicom_fails))
    return sorted(list(out_directories))
            
def dicom_series_to_nifti(dicom_files, orientation='LPS', quiet=False, json=False):
    ''' Convert a list of dicom_files to NiFTi. 
    
    Parameters
    ----------
    dicom_files : list
        List of dicom filenames
    orientation : Optional[str]
        An orientation string or tuple of 3 characters
        
    Returns
    -------
    tuple
        A tuple, with a nibabel.Nifti1Image instance as the first element. If diffusion bval/bvecs were found, they are included. If json, path to json file is included. 
    '''
    # delete this function when merged with pyDcm2nii
    logging.debug('diciphr.dicoms.dicom_series_to_nifti')
    try:
        from StringIO import StringIO
    except ImportError:  
        #python3  
        from io import StringIO
    try:
        dcm2nii_exe = which('dcm2niix')
    except:
        dcm2nii_exe = which('dcm2niix_afni')
    from glob import glob 
    try:
        tmpdir = make_temp_dir(prefix='dicom_series_to_nifti')
        dicom_files_copies = []
        for i,f in enumerate(dicom_files):
            dstfile=os.path.join(tmpdir, '{}.dcm'.format(i))
            # shutil.copyfile(os.path.realpath(os.readlink(f)), dstfile)
            shutil.copyfile(os.path.realpath(f), dstfile)
            dicom_files_copies.append(os.path.basename(dstfile))
        origDir=os.getcwd()
        os.chdir(tmpdir)
        cmd = [dcm2nii_exe, '-o', '.', '-f', 'tmp', '-x', 'n', '-z', 'y', '-t', 'n', '-s', 'n', '-i', 'n']
        if json:
            cmd.extend(['-b', 'y'])
        else:
            cmd.extend(['-b', 'n']) 
        if quiet:
            cmd.extend(['-v', '0'])
        else:
            cmd.extend(['-v', '1'])
        cmd.append(dicom_files_copies[0])  #have to shorten the filenames so dcm2nii will run. 
        returncode, stdout, stderr = ExecCommand(cmd, quiet=quiet).run()
        os.chdir(origDir)
        nifti_file = os.path.join(tmpdir, 'tmp.nii.gz')
        if not os.path.exists(nifti_file):
            raise DiciphrException('Nifti file was not created by {}'.format(dcm2nii_exe))
        bval_file = strip_nifti_ext(nifti_file)+'.bval'
        bvec_file = strip_nifti_ext(nifti_file)+'.bvec'
        _diffusion = (os.path.exists(bval_file) and os.path.exists(bvec_file))
        if _diffusion:
            nifti_im, bvals, bvecs = read_dwi(nifti_file, bval_file, bvec_file)
            nifti_reor_im, bvals, bvecs = reorient_dwi(nifti_im, bvals, bvecs, orientation=orientation)
            ret = (nifti_reor_im, bvals, bvecs)
        else:
            nifti_im = read_nifti(nifti_file)
            nifti_reor_im = reorient_nifti(nifti_im, orientation=orientation)
            ret = (nifti_reor_im, ) 
        if json:
            json_file = os.path.join(tmpdir, 'tmp.json')
            json_obj = StringIO()
            with open(json_file, 'r') as fid:
                json_obj.writelines(fid.readlines())
            ret = ret + (json_obj, )
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
    finally:
        shutil.rmtree(tmpdir)
    return ret 
