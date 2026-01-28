# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, shutil, logging
import re
from glob import glob 
from collections import defaultdict
from diciphr.utils import which, find_all_files_in_dir, ExecCommand, TempDirManager
from diciphr.nifti_utils import ( read_nifti, read_dwi, write_nifti, 
            write_dwi, strip_nifti_ext, reorient_nifti, reorient_dwi )
from pydicom import dcmread 
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.datadict import dictionary_VR
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence
import pandas as pd

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
    return dcmread(filename, stop_before_pixels=stop_before_pixels, force=force)

def get_dicom_series_attributes(dicom_files, keys=None, replace_spaces=False, ignore_errors=True):
    logging.debug('diciphr.dicoms.get_dicom_series_attributes')
    if len(dicom_files) == 0:
        raise ValueError('Input sequence of dicom files is empty.')
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
            raise ValueError(f"No Nifti files were created by {dcm2nii_exe}")
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
            except:
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

def _sanitize_path_component(path):
    """
    Make a string safe for use as a single path component (directory/file path).
    Keeps letters, digits, '_', '-', '+', and '.'; everything else -> '_'.
    Collapses repeated underscores and strips leading/trailing whitespace/underscores.
    """
    s = ''.join(c if (c.isalnum() or c in ('_', '-', '+', '.')) else '_' for c in str(path))
    s = re.sub(r'_+', '_', s).strip(' _')
    return s or 'untitled'

def _empty_element(ds, tag_like):
    tag = Tag(tag_like)
    vr = dictionary_VR(tag)
    if vr is None:
        ds[tag] = DataElement(tag, 'UN', b'')
        return
    if vr == "SQ":
        ds[tag] = DataElement(tag, "SQ", Sequence([]))
    elif vr in {'OB', 'OD', 'OF', 'OL', 'OW', 'UN'}:
        ds[tag] = DataElement(tag, vr, b'')
    else:
        ds[tag] = DataElement(tag, vr, "")

def _scrub_all_date_time(ds):
    """Remove all DA/TM/DT VR fields recursively."""
    for tag in list(ds.keys()):
        elem = ds[tag]
        vr = elem.VR
        if vr == "SQ":
            for item in elem:
                _scrub_all_date_time(item)
            continue

        if vr in ("DA", "TM", "DT"):
            ds[tag].value = ""

def _build_tag_list(anonymization_level):
    """
    Returns a list of tags to empty (zero-length).
    Extends your list depending on anonymization_level.
    """
    # Base identity PHI 
    basic_tags = [
        (0x0008,0x0050),  # AccessionNumber
        (0x0008,0x0080),  # InstitutionName
        (0x0008,0x0081),  # InstitutionAddress
        (0x0008,0x0090),  # ReferringPhysicianName
        (0x0008,0x0092),  # ReferringPhysicianAddress
        (0x0008,0x0094),  # ReferringPhysicianTelephone
        (0x0008,0x1010),  # StationName
        (0x0008,0x1040),  # InstitutionalDepartmentName
        (0x0008,0x1048),  # PhysiciansOfRecord
        (0x0008,0x1050),  # PerformingPhysicianName
        (0x0008,0x1060),  # NameOfPhysiciansReadingStudy        
        (0x0008,0x1072),  # OperatorsIdentificationSequence
        (0x0008,0x1070),  # OperatorsName
        (0x0008,0x1080),  # AdmittingDiagnosesDescription        
        (0x0010,0x0010),  # PatientName
        (0x0010,0x0020),  # PatientID
        (0x0010,0x0030),  # PatientBirthDate
        (0x0010,0x0032),  # PatientBirthTime
        (0x0010,0x0040),  # PatientSex
        (0x0010,0x1000),  # OtherPatientIDs
        (0x0010,0x1001),  # OtherPatientNames
        (0x0010,0x1010),  # PatientAge
        (0x0010,0x1020),  # PatientSize
        (0x0010,0x1030),  # PatientWeight
        (0x0010,0x1090),  # MedicalRecordLocator
        (0x0010,0x2150),  # CountryOfResidence
        (0x0010,0x2154),  # RegionOfResidence
        (0x0010,0x2160),  # EthnicGroup
        (0x0010,0x2180),  # Occupation
        (0x0010,0x21A0),  # SmokingStatus
        (0x0010,0x21B0),  # AdditionalPatientHistory
        (0x0010,0x21C0),  # PregnancyStatus
        (0x0010,0x21F0),  # PatientReligiousPreference
        (0x0010,0x4000),  # PatientComments
        (0x0020,0x0010),  # StudyID
        (0x0020,0x4000),  # ImageComments
        (0x0032,0x1020),  # Requesting Physician Identification Sequence
        (0x0032,0x1032),  # Requesting Physician
        (0x0032,0x1033),  # Requesting Service
        (0x0032,0x1060), # Requested Procedure Module
        (0x0040,0x0242),  # PerformedProcedureStepID
        (0x0040,0x0253),  # Performed Procedure Step ID
        (0x0040,0x0254), # Performed Procedure Step Description
        (0x0040,0x0275),  # RequestAttributesSequence
        (0x0040,0x0280), # Comments on the Performed Procedure Step
    ]
    # More extensive PHI needed for moderate anonymization
    moderate_extra = [
        (0x0018,0x1002),  # DeviceUID
        (0x0018,0x700A),  # DetectorID
        (0x0070,0x0001),  # GraphicAnnotationSequence
        (0x0070,0x0004),  # TextObjectSequence
        (0x0070,0x0081),  # ContentLabel
        (0x0070,0x0082),  # ContentDescription
        (0x0070,0x0084),  # ContentCreatorsName
        (0x0070,0x0086),  # ContentCreatorsIdentificationCodeSequence
        (0x0070,0x0086),  # ContentCreatorIdentificationCodeSequence
        (0x0018,0x1000),  # DeviceSerialNumber
        (0x0018,0x1020),  # SoftwareVersions
        (0x0008,0x0082),  # InstitutionCodeSequence
        (0x0008,0x1018),  # DeviceID
        (0x0008,0x0052),  # QueryRetrieveLevel
    ]
    # “Strict” removes nearly everything possibly PHI:
    strict_extra = [
        (0x0008,0x0070),  # Manufacturer
        (0x0008,0x1090),  # ManufacturerModelName
        (0x0008,0x1030),  # StudyDescription
    ]
    
    if anonymization_level == "basic":
        return basic_tags
    elif anonymization_level == "moderate":
        return basic_tags + moderate_extra
    elif anonymization_level == "strict":
        return basic_tags + moderate_extra + strict_extra
    else:
        raise ValueError("anonymization_level must be 'basic', 'moderate', or 'strict'")

def _remove_private_tags_except_siemens_csa(ds):
    """
    Remove private tags from a DICOM dataset with Siemens CSA preservation logic.

    Behavior:
      - If Siemens CSA tags are present, preserve:
          * Siemens CSA elements (0029,1010) and (0029,1020)
          * All private elements in Siemens groups 0x0019 and 0x0029
          * The corresponding private creator elements in those groups (gggg,00xx)
        Remove all other private tags (recursively, including in sequences).

      - If Siemens CSA tags are absent, perform a standard recursive
        ds.remove_private_tags() to strip all private elements.

    Notes:
      * This function modifies the dataset in-place.
      * It mirrors pydicom's recursive behavior for sequences.
    """
    SIEMENS_CSA_TAGS = {
        Tag(0x0029, 0x1010),  # CSA Image Header Info
        Tag(0x0029, 0x1020),  # CSA Series Header Info
    }
    # Common Siemens private groups that frequently contain diffusion info
    SIEMENS_KEEP_GROUPS = {0x0019, 0x0029}
    if not any(tag in ds for tag in SIEMENS_CSA_TAGS):
        # No CSA → do the standard, fully-recursive removal.
        ds.remove_private_tags()
        return
    # CSA present → selective removal:
    # We traverse recursively and collect tags to delete so we don't mutate while iterating.
    def _prune_private_recursive(dset: Dataset):
        to_delete = []
        for elem in dset:
            tag = elem.tag
            # Recurse into sequences
            if elem.VR == "SQ":
                for item in elem:
                    _prune_private_recursive(item)
                continue
            # Only consider private elements for deletion
            if not tag.is_private:
                continue
            g = tag.group
            e = tag.element
            # Keep: Siemens CSA tags explicitly
            if tag in SIEMENS_CSA_TAGS:
                continue
            # Keep: any private element in Siemens keep-groups (0019, 0029)
            if g in SIEMENS_KEEP_GROUPS:
                continue
            # Also keep: private creator elements for Siemens groups we keep:
            # Private creator slots are (gggg,00xx) with 0x0010 <= xx <= 0x00FF
            if (e & 0xFF00) == 0x0000 and 0x0010 <= (e & 0x00FF) <= 0x00FF:
                # Only keep creator elements if they belong to a Siemens keep-group
                # (i.e., creator for 0019 or 0029). Creators are per-group, so g matters.
                if g in SIEMENS_KEEP_GROUPS:
                    continue
            # All other private elements should be removed
            to_delete.append(tag)

        for tag in to_delete:
            # It's safe to ignore KeyErrors if already removed by parent logic
            if tag in dset:
                del dset[tag]
    _prune_private_recursive(ds)

def anonymize_dicomfile(infile, outfile, anonymization_level='moderate'):
    if not is_dicom_file(infile):
        raise InvalidDicomError(f"{infile} is not a DICOM file")
    ds = dcmread(infile, force=True)
    # 1. Scrub ALL date/time fields (DA/TM/DT)
    _scrub_all_date_time(ds)
    # 2. Scrub PHI tag list depending on level
    tag_list = _build_tag_list(anonymization_level)
    for tag in tag_list:
        if tag in ds:
            _empty_element(ds, tag)
    # 3. Remove ALL private tags
    _remove_private_tags_except_siemens_csa(ds)
    # 4. Generate a new SOPInstanceUID (required if modified)
    ds.SOPInstanceUID = generate_uid()
    ds.save_as(outfile)
    return outfile

def run_dicom_to_nifti(subject, dicom_dir, nifti_dir, sort_mode='none', dicom_sort_dir=None, 
                       no_convert=False, orientation='LPS', decompress=False, 
                       encode_dates=False, anonymization_level='moderate'):
    logging.info(f'Subject: {subject}')
    logging.info(f'DICOM directory: {dicom_dir}')
    logging.info(f'NIfTI output directory: {nifti_dir}')
    sort_mode=str(sort_mode).lower()[:4]
    if sort_mode not in ('none', 'link', 'copy', 'move', 'anon'):
        raise ValueError("sort_mode must be one of: 'none', 'link', 'copy', 'move', 'anonymize'")
    if sort_mode != 'none':
        if dicom_sort_dir is None:
            raise ValueError("If sort_mode is not 'none', dicom_sort_dir must be provided")
        os.makedirs(dicom_sort_dir, exist_ok=True)
        logging.info(f'Sorted DICOMs directory: {dicom_sort_dir}')
        if sort_mode == 'anon' and encode_dates is False:
            encode_dates = True
            logging.info("Anonymization mode: setting encode_dates to True")
    # initialize attributes dataframe
    df = pd.DataFrame(columns=['Subject']+default_keys+['Nifti'])
    dicom_nifti_map = {}
    # begin 
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
            # ---- optional sorting step ----
            path_string = f"{_sanitize_path_component(studydate)}_s{seriesnum:03d}_{_sanitize_path_component(seriesdesc)}"
            if sort_mode != 'none':
                # Construct group directory name and create it
                dest_dir = os.path.join(dicom_sort_dir, path_string)
                os.makedirs(dest_dir, exist_ok=True)
                # Fan-in files into the group dir (symlink/copy/move/anon)
                dest_paths = []
                for i, src in enumerate(dicom_files):
                    # Prefix with an index to prevent name collisions across scanners/folders
                    dst = os.path.join(dest_dir, f"{i:05d}_" + os.path.basename(src))
                    if sort_mode == 'link':
                        if not os.path.exists(dst):
                            try:
                                os.symlink(os.path.realpath(src), dst)
                            except OSError:
                                # Fallback if symlink not permitted (e.g., Windows w/o privileges)
                                shutil.copy2(src, dst)
                    elif sort_mode == 'copy':
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                    elif sort_mode == 'move':
                        if not os.path.exists(dst):
                            # If destination exists (idempotent rerun), skip moving to avoid overwrite
                            shutil.move(src, dst)
                    elif sort_mode == 'anon':
                        if not os.path.exists(dst):
                            # If destination exists (idempotent rerun), skip moving to avoid overwrite
                            anonymize_dicomfile(src, dst, anonymization_level=anonymization_level)
                    dest_paths.append(dst)
                dicom_files_for_conversion = dest_paths
            else:
                dicom_files_for_conversion = dicom_files
            # ---- end sorting step ----
            row = pd.Series(attributes)
            row['Subject'] = subject    
            if no_convert is False:
                output_prefix = os.path.join(nifti_dir, subject) + "_" + path_string
                logging.info(f"Convert dicoms to nifti file {output_prefix}")
                nifti_files = dicom_series_to_nifti(
                        dicom_files_for_conversion, output_prefix,
                        orientation=orientation,
                        quiet=True, json=True, decompress=decompress
                    )
                row['Nifti'] = ' '.join(nifti_files)
                dicom_nifti_map[nifti_files[0]] = dicom_files_for_conversion
            df = df.append(row, ignore_index=True)
        except Exception:
            logging.exception(f'Failed to convert DICOMs for StudyInstanceUID,SeriesInstanceUID {uid_key}')
    logging.info('Conversion complete.')
    return df, dicom_nifti_map