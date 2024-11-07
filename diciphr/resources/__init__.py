# -*- coding: utf-8 -*-

import os as _os

_resources_dir = _os.path.dirname(_os.path.realpath(__file__))

_labels_86_textfile = _os.path.join(_resources_dir, '86_labels.txt')
labels_86 = [int(a) for a in open(_labels_86_textfile,'r').readlines()]

_labels_87_textfile = _os.path.join(_resources_dir, '87_labels.txt')
labels_87 = [int(a) for a in open(_labels_87_textfile,'r').readlines()]

labels_wm = [2,7,41,46,250,251,252,253,254,255]

desikan86_nodes_csv = _os.path.join(_resources_dir, 'Desikan_86_lut.csv')
desikan87_nodes_csv = _os.path.join(_resources_dir, 'Desikan_87_lut.csv')

desikan86_assignments_function = _os.path.join(_resources_dir, 'assignments_function.csv')
desikan86_assignments_cognition = _os.path.join(_resources_dir, 'assignments_cognition.csv')

