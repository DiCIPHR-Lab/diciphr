from ..utils import DiciphrException
from .connmat_utils import normalize_mat
from collections import OrderedDict
import numpy as np

desikan_cognitive_systems = OrderedDict((
    ("Motor",[15,20,22,58,63,65,35,78]),
    ("Visual",[3,5,7,9,11,19,46,48,50,52,54,62,27,70,14,57]),
    ("Language",[16,18,28,29,6,0,17,25,13]),
    ("Attention",[27,70,26,69,29,72,16,59,17,60,18,61,25,68]),
    ("Memory",[39,82,4,47,14,57,35,78,42,85]),
    ("Social Cognition",[40,83,10,53,48,76,71,39,82,42,85]),
    ("Reward",[41,84,36,79,37,80,42,85,40,83,38,81]),
    ("Cognitive Control",[22,65,2,45,1,44,30,73,10,53,12,55,16,59,17,60,18,61,24,67,25,68,26,69])
))
desikan_functional_systems = OrderedDict((
    ("Auditory",[0,28,32,43,71,75]),
    ("Visual",[3,4,5,7,9,11,19,46,47,48,50,52,54,62]),
    ("Motor and Somatosensory",[15,20,22,58,63,65]),
    ("Default Mode",[8,21,23,26,51,64,66,69]),
    ("Cingulo-Opercular",[1,16,17,24,25,29,44,59,60,67,68,72]),
    ("Fronto-Parietal",[2,6,12,18,30,33,45,49,55,61,73,76]),
    ("Dorsal Attention",[27,70]),
    ("Ventral Attention",[10,53]),
    ("Subcortical",[35,36,37,38,39,40,41,42,78,79,80,81,82,83,84,85]),
    ("Cerebellum",[34,77]),
    ("Other",[13,14,31,56,57,74])
))

atlasppatt_chen_systems = OrderedDict((
    ('Somatomotor', [22,23,24,25,27,28,31,32,33,34,35,36,37,48,49,50,52,111,119,120,121,122,123,124,125,149,150,151,152,154,155,156,157,158,159,160,161,162,163,164,171,234,242,243,244,245,246,247,248]),
    ('Saccade', [11,13,14,21,23,26,27,28,55,57,58,67,119,120,121,122,123,124,125,139,141,142,147,149,153,154,155,173,175,179,242,243,244,245,246,247,248,249,250,251]),
    ('Vergence', [57,58,66,67,74,75,76,77,119,120,121,122,123,124,125,179,186,187,200,201,202,203,242,243,244,245,246,247,248,249,250,251]),
    ('Vestibular', [11,26,27,28,45,46,47,48,49,50,51,57,58,66,67,106,107,116,119,120,121,122,123,124,125,139,153,154,155,170,171,172,173,174,178,179,186,187,230,231,239,242,243,244,245,246,247,248])
))

atlasppatt_yeo_systems = OrderedDict((
    ('Visual', [58,59,65,66,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,87,119,187,188,189,199,200,201,202,203,204,205,206,207,208,209,210,212,242]),
    ('Somatomotor', [32,33,34,35,36,37,44,45,46,47,48,52,99,100,101,104,105,120,150,155,156,157,158,160,163,164,170,171,172,173,174,175,176,177,227,228,230,232,233,243]),
    ('Dorsal Attention', [31,49,53,54,55,56,57,70,93,121,151,154,159,161,181,182,183,184,185,186,194,211,219,220,244]),
    ('Ventral Attention', [9,23,24,25,30,38,40,41,51,98,102,106,107,122,138,149,162,166,168,178,179,221,245]),
    ('Limbic', [0,3,5,7,8,86,88,89,90,91,123,126,129,131,133,213,215,216,217,218,231,246]), 
    ('Frontoparietal', [2,10,11,12,13,14,15,16,18,26,27,28,29,42,50,60,62,92,94,124,128,135,137,139,141,153,167,193,247]), 
    ('Default', [1,4,6,17,19,20,21,22,39,43,61,63,64,67,68,69,95,96,97,103,125,127,130,132,134,136,140,142,143,144,145,146,147,148,152,165,169,180,190,191,192,195,196,197,198,214,222,223,224,225,226,229,248]), 
    ('Subcortical', [108,109,110,111,112,113,114,115,116,117,118,234,235,236,237,238,239,240,241,249,250,251]),
    ('Attention', [31,49,53,54,55,56,57,70,93,121,151,154,159,161,181,182,183,184,185,186,194,211,219,220,244,9,23,24,25,30,38,40,41,51,98,102,106,107,122,138,149,162,166,168,178,179,221,245])
))

def get_system(atlas='desikan', system='cognitive'):
    if atlas == 'desikan':
        if system.lower() == 'cognitive':
            return desikan_cognitive_systems
        elif system.lower() == 'functional':
            return desikan_functional_systems
        else:
            raise DiciphrException('Desikan system name not recognized: {}'.format(system))
    elif atlas == 'atlasppatt':
        if system.lower() == 'chen':
            return atlasppatt_chen_systems
        elif system.lower() == 'yeo':
            return atlasppatt_yeo_systems
        else:
            raise DiciphrException('Atlasppatt system name not recognized: {}'.format(system))
    else:
        raise DiciphrException('Atlas name not recognized: {}'.format(atlas))

def systemConnectivity(A, systemsDict, normalize=False, upper_triangular=False):
    if normalize:
        A = normalize_mat(A)
    systems = systemsDict.values()
    numSys = len(systems)
    Conn = np.zeros([numSys,numSys])
    #compute within and between system mean connectivity
    for i in range(numSys):
        #get the first system indices
        idx1 = systems[i]
        for j in range(i, numSys):
            #get the second system indices
            idx2 = systems[j]
            #get connections between systems
            subA = A[idx1,:][:,idx2]
            #if two systems are same, diagonal will be zero and mean will be missleading
            #instead, first get the upper triangular and then calculate the mean
            if (i==j):
                upper = np.triu_indices(len(idx1), 1)
                subA = subA[upper]
            Conn[i,j] = subA.mean()
            Conn[j,i] = Conn[i,j]
    #now vectorize connectivity
    #order of the vector:
    #s1-s1, s1-s2, s1-s3, ..., s2-s2, s2-s3, ...
    upper = np.triu_indices(numSys, 0)
    if upper_triangular:
        return Conn[upper]
    else:
        return Conn

