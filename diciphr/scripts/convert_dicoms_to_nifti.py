#! /usr/bin/env python
import os, sys, shutil, logging, traceback, datetime, argparse
import pandas as pd
from ..nifti_utils import ( write_nifti, write_dwi, strip_nifti_ext )
from ..dicoms import ( dicom_series_to_nifti, get_dicom_series_attributes,
                sort_and_link_dicoms_by_series_attributes)
from ..utils import ( find_all_files_in_dir, make_dir, 
                check_inputs, protocol_logging ) 

DESCRIPTION = '''
    Converts dicoms sorted by subject to LPS Nifti files.
'''

PROTOCOL_NAME='convert_dicoms_to_nifti'    
file_template = '{subject}_{date}_s{seriesnumber:03d}_{series}.nii.gz'
index_template = '{subject}:s{seriesnumber:03d}_{series}'

def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-s',action='store',metavar='subject',dest='subjects',
                    type=str, required=True, 
                    help='A text file of subject IDs, or a single subject ID'
                    )
    p.add_argument('-o', action='store', metavar='outdir', dest='project_dir', 
                    type=str, required=False, default=os.getcwd(), 
                    help='The project directory. The default is {}.'.format(os.getcwd())
                    )
    p.add_argument('-c', action='store', metavar='csvfile', dest='csv_file',
                    type=str, required=False, default=None,
                    help='Name of csv file to log information about each Nifti converted. The default is "{project_dir}/Lists/convert_dicoms_to_nifti_(date)-(time).csv".'
                    )
    p.add_argument('-d', action='store', metavar='dcmdir', dest='dicoms_dir', 
                    type=str, required=False, default=None, 
                    help='Name of the subdirectory of {project_dir} where dicoms are located. Default is "dicoms".'
                    )
    p.add_argument('-r', '--orient', action='store', metavar='orn', dest='orientation', 
                    type=str, required=False, default='LPS',
                    help='Orientation of the output nifti images. Default is LPS.'
                    )
    p.add_argument('-n', '--no-convert', action='store_true', dest='no_convert',
                    required=False, default=False, 
                    help='Only sort dicoms and make links, do not convert to Nifti.'
                    )
    p.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    return p

def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    project_dir = os.path.realpath(args.project_dir)
    try:
        check_inputs(project_dir, directory=True)
        if args.dicoms_dir is None:
            dicoms_dir = os.path.join(project_dir, 'dicoms')
        else:
            dicoms_dir = os.path.realpath(args.dicoms_dir)
        check_inputs(dicoms_dir, directory=True)
        nifti_dir = os.path.join(project_dir, 'Nifti')
        logs_dir = os.path.join(nifti_dir, 'logs')
        lists_dir = os.path.join(project_dir, 'Lists')
        make_dir(nifti_dir, recursive=False, pass_if_exists=True)
        make_dir(logs_dir, recursive=False, pass_if_exists=True)
        make_dir(lists_dir, recursive=False, pass_if_exists=True)
        log_file = protocol_logging(PROTOCOL_NAME, logs_dir, debug=args.debug)
        if args.csv_file is None:
            csv_file = os.path.join(lists_dir, os.path.basename(log_file[:-4] + '.csv')) #strip .log off log file  
        else:
            csv_file = os.path.realpath(args.csv_file)
        check_inputs(os.path.dirname(csv_file), directory=True)
        if os.path.isfile(args.subjects):
            subjects = [ _line.strip() for _line in open(args.subjects, 'r').readlines() ]
        else:
            subjects = [args.subjects]
        for subject in subjects:
            subject_dicom_dir = os.path.join(dicoms_dir, subject)
            check_inputs(subject_dicom_dir, directory=True)
        for subject in subjects:
            subject_dicom_dir = os.path.join(dicoms_dir, subject)
            subject_nifti_dir = os.path.join(nifti_dir, subject)
            make_dir(subject_nifti_dir, recursive=False, pass_if_exists=True)
            run_dicom_to_nifti(subject, subject_dicom_dir, subject_nifti_dir, csv_file, orientation=args.orientation, no_convert=args.no_convert)
    except Exception as e:
        logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise e
        
def run_dicom_to_nifti(subject, dicom_dir, nifti_dir, csv_file, orientation='LPS', no_convert=False):
    logging.info('Subject: {}'.format(subject))
    logging.info('dicom_dir: {}'.format(dicom_dir))
    logging.info('nifti_dir: {}'.format(nifti_dir))
    logging.info('csv_file: {}'.format(csv_file))
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame() #create a new csv
    logging.info('Begin')
    logging.info('Finding dicom files')
    dicoms_sorted_dir = os.path.join(nifti_dir, 'dicoms_sorted')
    make_dir(dicoms_sorted_dir, recursive=False, pass_if_exists=True)
    dicom_files = find_all_files_in_dir(dicom_dir)
    dicom_directories = sort_and_link_dicoms_by_series_attributes(dicom_files, dicoms_sorted_dir)
    if no_convert:
        logging.info('Skipping conversion to Nifti.')
    else:
        logging.info('Looping through dicom series')
        for directory in dicom_directories:
            try:
                dicom_files = find_all_files_in_dir(directory)
                dicom_attributes = get_dicom_series_attributes(dicom_files, replace_spaces=True)
                #attributes to fill in filename
                try:
                    dicom_date = dicom_attributes['AcquisitionDate'][0]#.replace(' ','_')
                except:
                    logging.warning('Could not get dicom date.')
                    dicom_date = ''
                try:
                    dicom_series = dicom_attributes['SeriesDescription'][0]#.replace(' ','_')
                except:
                    logging.warning('Could not get dicom series description.')
                    dicom_series = ''
                try:
                    dicom_seriesnumber = int(dicom_attributes['SeriesNumber'][0])
                except:
                    logging.warning('Could not get dicom series number.')
                    dicom_seriesnumber = 0
                filename_out = os.path.join(nifti_dir, file_template.format(
                    subject=subject,
                    date=dicom_date,
                    seriesnumber=dicom_seriesnumber,
                    series=dicom_series,
                    ))
                json_filename_out = strip_nifti_ext(filename_out) + ".json"
                index_string = index_template.format(
                    subject=subject,
                    seriesnumber=dicom_seriesnumber,
                    series=dicom_series
                )
                #log info about the dicom series
                logging.info('Subject: {}'.format(subject))
                logging.info('directory: {}'.format(directory))
                df.loc[index_string,'SubjectID'] = subject
                for key, value in dicom_attributes.items():
                    value_as_str = list(map(str,value))
                    if len(set(value_as_str)) >= 1:
                        value_as_str = value_as_str[0]
                    elif len(set(value_as_str)) == 0:
                        value_as_str = 'NA'
                    
                    df.loc[index_string,key] = value_as_str
                #convert to nifti
                logging.debug('dicom_files: {}'.format(dicom_files))
                nifti_converted = dicom_series_to_nifti(dicom_files, orientation=orientation, quiet=True, json=True)
                if len(nifti_converted) == 2:
                    nifti_im, json_obj = nifti_converted
                    write_nifti(filename_out, nifti_converted[0])
                    logging.info('Wrote Nifti to file {}'.format(filename_out))
                elif len(nifti_converted) == 4:
                    dwi_im, bvals, bvecs, json_obj = nifti_converted
                    write_dwi(filename_out, dwi_im, bvals, bvecs)
                    logging.info('Wrote Diffusion Nifti to file {}'.format(filename_out))
                with open(json_filename_out, 'w') as fid:
                    json_obj.seek(0)
                    shutil.copyfileobj(json_obj, fid)
                #log to csv_file
                df.loc[index_string, dicom_series] = filename_out
            except Exception as e:
                logging.error('Could not convert dicoms in directory {} to nifti.'.format(directory))
                logging.error(''.join(traceback.format_exception(*sys.exc_info())))
        logging.info('Saving csv to file {}'.format(csv_file))
        df.to_csv(csv_file, index=False)
    logging.info('Done')

if __name__ == '__main__':
    main(sys.argv[1:])