#! /usr/bin/env python
import os, sys, logging
from diciphr.utils import ( check_inputs, make_dir, protocol_logging,
                    DiciphrArgumentParser )
from diciphr.nifti_utils import read_nifti, nifti_image, write_nifti
import numpy as np
import pandas as pd

DESCRIPTION = '''
    Replace labels in an atlas with a new list. 
'''

PROTOCOL_NAME='replace_labels'    
    
def buildArgsParser():
    p = DiciphrArgumentParser(description=DESCRIPTION)
    p.add_argument('-a',action='store',metavar='atlas',dest='atlas',
                    type=str, required=True, 
                    help='The atlas image in Nifti format.'
                    )
    p.add_argument('-o',action='store',metavar='output',dest='output',
                    type=str, required=True, 
                    help='The output Nifti filename.'
                    )
    p.add_argument('-c',action='store',metavar='csv',dest='csv',
                    type=str, required=False, default=None,
                    help='A lookup csv. The first column should be the input labels, the second column should be the output labels.'
                    )
    p.add_argument('-l',action='store',metavar='list',dest='input_list',
                    type=str, required=False, default=None,
                    help='The input labels as a text file.'
                    )
    p.add_argument('-m',action='store',metavar='list',dest='output_list',
                    type=str, required=False, default=None,
                    help='The output labels as a text file.'
                    )
    p.add_argument('--order', action='store_true', dest='order', 
                    required=False, default=False, 
                    help='Replace the atlas with an ordered version of its labels.'
                    )
    return p
    
def main(argv):
    parser = buildArgsParser()
    args = parser.parse_args(argv)
    output_dir = os.path.dirname(os.path.realpath(args.output))
    make_dir(output_dir, recursive=True, pass_if_exists=True)
    protocol_logging(PROTOCOL_NAME, directory=args.logdir, filename=args.logfile, debug=args.debug, create_dir=True)
    try:
        check_inputs(args.atlas, nifti=True)
        atlas_im = read_nifti(args.atlas)
        if args.csv:
            logging.info('CSV lookup mode.')
            check_inputs(args.csv)
            lut = pd.read_csv(args.csv, header=None)
            columns = [lut.columns[0], lut.columns[1]]
            lut = lut.loc[:,columns]
            lut = lut.to_numpy()
            input_list = list(lut[:,0])
            output_list= list(lut[:,1])
        elif args.output_list:
            logging.info('Input list/output list mode.')
            check_inputs(args.input_list)
            input_list = list([float(a.strip()) for a in open(args.input_list,'r').readlines()])
            check_inputs(args.output_list)
            output_list = list([float(a.strip()) for a in open(args.output_list,'r').readlines()])
        elif args.input_list:
            logging.info('Input list mode.')
            if not args.order:
                raise ValueError('No output list given and order is False. Nothing to do!')
            check_inputs(args.input_list)
            input_list = list([int(a.strip()) for a in open(args.input_list,'r').readlines()])
            output_list = range(1,1+len(input_list))
        elif args.order:
            _data = atlas_im.get_fdata()
            input_list = list(np.unique(_data[_data>0]))
            output_list = range(1,1+len(input_list))
        else:
            raise ValueError('No output list given and order_labels is False. Nothing to do!')
        logging.info('Run replace_labels')
        replace_labels(atlas_im, args.output, input_list, output_list)
    except Exception:
        logging.exception(f"Exception encountered running {PROTOCOL_NAME}")
        raise
    
def replace_labels(atlas_im, output_filename, input_list, output_list):
    atlas_data = atlas_im.get_fdata()
    atlas_affine = atlas_im.affine
    logging.info('Input list: '+', '.join(map(str,input_list)))
    logging.info('Output list: '+', '.join(map(str,output_list)))
    new_atlas_data = np.zeros(atlas_data.shape, dtype=np.float32)
    for lbl, val in zip(input_list, output_list):
        new_atlas_data[atlas_data == lbl] = val 
    new_atlas_im = nifti_image(new_atlas_data, atlas_affine)
    logging.info(f'Save to output {output_filename}')
    write_nifti(output_filename, new_atlas_im)
    
if __name__ == '__main__': 
    main(sys.argv[1:])
