#! /usr/bin/env python
import sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from diciphr.oscar import Oscar, oscar_argparser, run_oscar_commandline

DESCRIPTION='''OSCAR – Utility to Overlay Statistical Content on Anatomical Reference - visualizes neuroimaging and/or statistical results as static or animated composites'''

def main(argv):
    parser = oscar_argparser()
    args = parser.parse_args(argv)
    run_oscar_commandline(args)
    
if __name__ == '__main__':
    main(sys.argv[1:])
