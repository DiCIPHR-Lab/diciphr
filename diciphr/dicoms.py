# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, shutil, logging
from datetime import datetime
from glob import glob 
from collections import defaultdict
from diciphr.utils import ( which, find_all_files_in_dir, 
            ExecCommand, TempDirManager, DiciphrException )
from diciphr.nifti_utils import ( read_nifti, read_dwi, write_nifti, 
            write_dwi, strip_nifti_ext, reorient_nifti, reorient_dwi )
import pydicom 
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
import pandas as pd
from pandas.errors import EmptyDataError

#############################################
############  DICOM Utilities  ##############
#############################################
hex_lookup = {
    'SeriesInstanceUID':0x0020000e,
    'PatientID':0x00100020,
    'SeriesNumber':0x00200011,
    'PixelSpacing':0x00280030,
    'SliceThickness':0x00180050,
    'SpacingBetweenSlices':0x00180088,
    'Rows':0x00280010,
    'Columns':0x00280011,
    'RepetitionTime':0x00180080,
    'EchoTime':0x00180081,
    'InversionTime':0x00180082,
    'DwellTime':0x00191018,
    'FlipAngle':0x00181314,
    'PixelBandwidth':0x00180095, 
    'BandwidthPerPixelPhaseEncode':0x00191028,
    'NumberPhaseEncodingSteps':0x00180089, 
    'InPlanePhaseEncodingDirection':0x00181312,
    'MagneticFieldStrength':0x00180087,
    'ProtocolName':0x00181030,
    'SeriesDescription':0x0008103e,
    'Manufacturer':0x00080070,
    'ManufacturerModelName':0x00081090,
    'DeviceSerialNumber':0x00181000,
    'InstitutionName':0x00080080,
    'StudyDate':0x00080020,
    'StudyInstanceUID':0x0020000d,
    'SeriesDate':0x00080021,
    'AcquisitionDate':0x00080022,
    'SeriesTime':0x00080031,
    'ContentDate':0x00080023
}

default_keys = [
    'PatientID',
    'StudyInstanceUID',
    'StudyDate',
    'InstitutionName',
    'Manufacturer',
    'ManufacturerModelName',
    'DeviceSerialNumber', 
    'MagneticFieldStrength',
    'SeriesInstanceUID',
    'SeriesNumber',
    'SeriesDescription',
    'SeriesDate', 
    'SeriesTime',
    'Rows', 
    'Columns',
    'PixelSpacing',
    'SpacingBetweenSlices',
    'SliceThickness',
    'RepetitionTime',
    'EchoTime',
    'InversionTime',
    'DwellTime', 
    'FlipAngle',
    'PixelBandwidth', 
    'BandwidthPerPixelPhaseEncode',
    'NumberPhaseEncodingSteps',
    'InPlanePhaseEncodingDirection'
]

def is_dicom_file(filepath, force=False):
    """
    Checks if the given file is a valid readable DICOM file.

    Parameters:
        filepath (str): Path to the file to check.

    Returns:
        bool: True if the file is a DICOM file, False otherwise.
    """
    try:
        read_dicom_file(filepath, stop_before_pixels=True, force=force)
        return True
    except (InvalidDicomError, FileNotFoundError, IsADirectoryError, PermissionError):
        return False

def read_dicom_file(filename, stop_before_pixels=False, force=False):
    logging.debug('diciphr.dicoms.read_dicom_file')
    return pydicom.dcmread(filename, stop_before_pixels=stop_before_pixels, force=force)

def get_dicom_series_attributes(dicom_files, keys=None, replace_spaces=False, ignore_errors=True):
    logging.debug('diciphr.dicoms.get_dicom_series_attributes')
    if len(dicom_files) == 0:
        raise DiciphrException('Input sequence of dicom files is empty.')
    if keys is None:
        keys = default_keys
    alt_keys = {
        'StudyDate':['SeriesDate','AcquisitionDate','ContentDate'],
        'SeriesDate':['StudyDate','AcquisitionDate','ContentDate'],
        'SeriesDescription':['ProtocolName'],
    }
    
    def get_dicom_attribute(dcm, filename, key, tried_keys=None, ignore_errors=True):
        if tried_keys is None:
            tried_keys = set()
        if key in tried_keys:
            return '' 
        tried_keys.add(key)
        hex_tag = hex_lookup[key]
        element = dcm.get(hex_tag)
        if element and element.value != '':
            return element.value
        for alt_key in alt_keys.get(key, []):
            value = get_dicom_attribute(dcm, filename, alt_key, tried_keys)
            if value != '':
                return value 
        all_keys = [key] + alt_keys.get(key, [])
        found_any = any(hex_lookup[k] in dcm for k in all_keys)
        if not ignore_errors and not found_any:
            raise KeyError(f"Attribute {key} and its alternates not found in DICOM dataset {filename}.")
        #elif not found_any:
        #    logging.warning(f"Attribute {key} and its alternates not found in DICOM dataset {filename}.")
        return '' 
        
    dicom_attributes = dict( (_k, []) for _k in keys )
    for dicom_file in dicom_files: 
        if os.path.basename(dicom_file) == 'DICOMDIR':
            logging.warning(f"Skipping dicom directory structure file {dicom_file}")
        elif os.path.basename(dicom_file) == 'PhoenixZIPReport':
            logging.warning(f"Skipping Siemens Phoenix report {dicom_file}")
        else:
            dcm = read_dicom_file(dicom_file, force=True, stop_before_pixels=True)
            for key in keys:
                _attribute = get_dicom_attribute(dcm, dicom_file, key)
                if isinstance(_attribute, MultiValue):
                    _attribute = ','.join(map(str, list(_attribute)))    
                if replace_spaces:
                    _attribute = '_'.join(str(_attribute).split())
                dicom_attributes[key].append(_attribute)
    return dicom_attributes

def _parse_time(time_str):
    for fmt in ("%H%M%S.%f", "%H%M%S"):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time format not recognized: {time_str}")

def group_dicoms_by_series_attributes(dicom_files, encode_dates=False):
    grouped_files = defaultdict(list)
    grouped_attributes = defaultdict(set)
    dicom_fails = []
    for dicom_file in dicom_files:
        if os.path.basename(dicom_file) in ['DICOMDIR', 'PhoenixZIPReport']:
            logging.warning(f"Skipping non-imaging file {dicom_file}")
            continue
        elif is_dicom_file(dicom_file):
            try:
                attrs = get_dicom_series_attributes([dicom_file], replace_spaces=True, ignore_errors=True)
                attrs = dict((k,v[0]) for k,v in attrs.items())
                key = (
                    attrs['StudyInstanceUID'],
                    attrs['SeriesInstanceUID']
                )
                grouped_files[key].append(dicom_file)
                grouped_attributes[key].add(tuple((ak,attrs[ak]) for ak in default_keys))
            except Exception as e:
                logging.warning(f"Error getting attributes for dicom file {dicom_file}: {e}")
                dicom_fails.append(dicom_file)
                raise 
        else:
            logging.warning(f"Path is not a valid dicom file {dicom_file}")
            dicom_fails.append(dicom_file)
    
    if dicom_fails:
        logging.warning(f"Could not process {len(dicom_fails)} files as DICOMs")

    return grouped_files, grouped_attributes

def dicom_series_to_nifti(dicom_files, output_prefix, decompress=False, json=True, orientation='LPS', quiet=False, raise_on_error=False):
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
    
    dcm2nii_exe = which('dcm2niix')
    gdcmconv_exe = which('gdcmconv')
    with TempDirManager(prefix='dcm2nii') as manager:
        tmpdir = manager.path()
        dicom_files_copies = []
        for i,f in enumerate(dicom_files):
            dstfile = os.path.join(tmpdir, f"{i}.dcm")
            if decompress:      
                decompress_cmd = [gdcmconv_exe, '-w', os.path.realpath(f), dstfile]
                ExecCommand(decompress_cmd, quiet=quiet).run(raise_on_error=raise_on_error)
            else:
                shutil.copyfile(os.path.realpath(f), dstfile)
            dicom_files_copies.append(dstfile)
        
        cmd = [dcm2nii_exe, '-o', tmpdir, '-f', 'tmp', '-x', 'n', '-z', 'y', '-t', 'n', '-s', 'n', '-i', 'n']
        if json:
            cmd.extend(['-b', 'y'])
        else:
            cmd.extend(['-b', 'n']) 
        if quiet:
            cmd.extend(['-v', '0'])
        else:
            cmd.extend(['-v', '1'])
        cmd.append(dicom_files_copies[0])  #have to shorten the filenames so dcm2nii will run. 
        ExecCommand(cmd, quiet=quiet).run(raise_on_error=raise_on_error)
        
        nifti_files = sorted(list(glob(os.path.join(tmpdir, 'tmp*.nii.gz'))))
        if not nifti_files:
            raise DiciphrException(f"Nifti file was not created by {dcm2nii_exe}")
        output_files = []
        for nifti_file in nifti_files:
            suffix = strip_nifti_ext(os.path.basename(nifti_file))[3:]
            if suffix:
                output_file = f"{output_prefix}{suffix}.nii.gz"
            else:
                output_file = f"{output_prefix}.nii.gz"
            try:
                nifti_im, bvals, bvecs = read_dwi(nifti_file)
                _diffusion=True
            except DiciphrException:
                nifti_im = read_nifti(nifti_file)
                _diffusion=False 
            if _diffusion:
                nifti_reor_im, bvals, bvecs = reorient_dwi(nifti_im, bvals, bvecs, orientation=orientation)
                logging.info(f"Write DWI file {output_file}")
                write_dwi(output_file, nifti_reor_im, bvals, bvecs)
            else:
                nifti_reor_im = reorient_nifti(nifti_im, orientation=orientation)
                logging.info(f"Write non-diffusion Nifti file {output_file}")
                write_nifti(output_file, nifti_reor_im)
            if json:
                json_file = strip_nifti_ext(nifti_file)+'.json'
                output_json_file = strip_nifti_ext(output_file)+'.json'
                shutil.copyfile(json_file, output_json_file)
            output_files.append(output_file)
    return output_files 

def run_dicom_to_nifti(subject, dicom_dir, nifti_dir, 
                       orientation='LPS', no_convert=False,
                       decompress=False, encode_dates=False):
    logging.info(f'Subject: {subject}')
    logging.info(f'DICOM directory: {dicom_dir}')
    logging.info(f'NIfTI output directory: {nifti_dir}')
    df = pd.DataFrame(columns=['Subject']+default_keys+['Nifti'])
    dicom_map = {}
    dicom_files = find_all_files_in_dir(dicom_dir)
    grouped_files, grouped_attributes = group_dicoms_by_series_attributes(
        dicom_files, encode_dates=encode_dates
    )
    if encode_dates:
        date_map={}
        all_dates = set()
        for attributes in grouped_attributes.values():
            attributes = dict(list(attributes)[0])
            all_dates.add(attributes['StudyDate'])
        all_dates = sorted(list(all_dates))
        for i, d in enumerate(all_dates):
            date_map.update({d:f't{i}' if encode_dates else d})
        logging.info(f"date_map: {date_map}")        
    for uid_key in grouped_files.keys():
        try:
            dicom_files = grouped_files[uid_key]
            attributes = grouped_attributes[uid_key]
            if len(attributes)>1:
                logging.warning(f"Encountered multiple sets of attributes for same StudyInstanceUID,SeriesInstanceUID {uid_key}, proceeding with first")
            attributes = dict(list(attributes)[0])
            seriesnum = int(attributes['SeriesNumber'])
            seriesdesc = attributes['SeriesDescription']
            if encode_dates:
                studydate = date_map[attributes['StudyDate']]
            else:
                studydate = attributes['StudyDate']
            output_prefix = os.path.join(nifti_dir, subject)
            output_prefix += f'_{studydate}_s{seriesnum:03d}_{seriesdesc}'
            logging.info(f"Convert dicoms to nifti file {output_prefix}")
            nifti_files = dicom_series_to_nifti(
                    dicom_files, output_prefix,
                    orientation=orientation,
                    quiet=True, json=True, decompress=decompress
                )
            row = pd.Series(attributes)
            row['Subject'] = subject
            row['Nifti'] = ' '.join(nifti_files)
            dicom_map[nifti_files[0]] = dicom_files
            df = df.append(row, ignore_index=True)
        except Exception:
            logging.exception(f'Failed to convert DICOMs for StudyInstanceUID,SeriesInstanceUID {uid_key}')
    logging.info('Conversion complete.')
    return df, dicom_map