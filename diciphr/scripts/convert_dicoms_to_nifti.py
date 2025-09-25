#! /usr/bin/env python
import os, sys, logging
from diciphr.nifti_utils import write_nifti, write_dwi, strip_nifti_ext
from diciphr.dicoms import run_dicom_to_nifti
from diciphr.utils import make_dir, check_inputs, protocol_logging, DiciphrArgumentParser

DESCRIPTION = '''
    Converts dicoms sorted by subject to LPS Nifti files.
'''

PROTOCOL_NAME='convert_dicoms'    

def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-s',action='store',metavar='subject',dest='subject',
                    type=str, required=True, 
                    help='The subject ID'
                    )
    p.add_argument('-d', action='store', metavar='dcmdir', dest='dicoms_dir', 
                    type=str, required=False, default=None, 
                    help='Name of the directory where dicoms are located. Default is {outdir}/dicoms/{subject}'
                    )
    p.add_argument('-o', action='store', metavar='outdir', dest='project_dir', 
                    type=str, required=False, default=os.getcwd(), 
                    help='The project directory. Directory "Nifti/{subject}" will be created inside. The default is '+os.getcwd()
                    )
    p.add_argument('-r', '--orient', action='store', metavar='orn', dest='orientation', 
                    type=str, required=False, default='LPS',
                    help='Orientation of the output nifti images. Default is LPS.'
                    )
    p.add_argument('-n', '--no-convert', action='store_true', dest='no_convert', required=False,
                    help='Only sort dicoms and make links, do not convert to Nifti.'
                    )
    p.add_argument('-x', '--decompress', action='store_true', dest='decompress', required=False, 
                    help='Decompress dicom data with gdcmconv before attempting to convert to Nifti.'
                    )
    p.add_argument('-t', '--encode-dates', action='store_true', dest='encode_dates', required=False,
                    help='If multiple dates are detected, append t0, t1, etc. to subject ID. If one date is detected, append nothing. Default behavior: appends scan date to filename.'
                    )
    return p

def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    project_dir = os.path.realpath(args.project_dir)
    check_inputs(project_dir, directory=True)
    nifti_dir = os.path.join(project_dir, 'Nifti')
    lists_dir = os.path.join(project_dir, 'Lists')
    logs_dir = args.logdir or os.path.join(project_dir, 'logs')
    log_file = protocol_logging(PROTOCOL_NAME, directory=logs_dir, filename=args.logfile, debug=args.debug, create_dir=True)
    csv_file = os.path.join(lists_dir, os.path.basename(log_file[:-4] + '_attributes.csv'))
    dicoms_map_file = os.path.join(lists_dir, os.path.basename(log_file[:-4] + '_dicompaths.txt'))
    try:
        if args.dicoms_dir is None:
            dicoms_dir = os.path.join(project_dir, 'dicoms', args.subject)
        else:
            dicoms_dir = os.path.realpath(args.dicoms_dir)
        check_inputs(dicoms_dir, directory=True)
        make_dir(nifti_dir, recursive=False, pass_if_exists=True)
        make_dir(lists_dir, recursive=False, pass_if_exists=True)
        check_inputs(os.path.dirname(csv_file), directory=True)
        subject_nifti_dir = os.path.join(nifti_dir, args.subject)
        make_dir(subject_nifti_dir, recursive=False, pass_if_exists=True)
        df, dicom_map = run_dicom_to_nifti(args.subject, dicoms_dir, subject_nifti_dir, 
                orientation=args.orientation, no_convert=args.no_convert, 
                decompress=args.decompress, encode_dates=args.encode_dates)
        df.to_csv(csv_file)
        with open(dicoms_map_file, 'w') as fid:
            for niftifile, dicomfiles in dicom_map.items():
                dicomfiles = ' '.join(dicomfiles)
                fid.write(f"{niftifile}: {dicomfiles}\n")
            
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
        
if __name__ == '__main__':
    main(sys.argv[1:])