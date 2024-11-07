import os, shutil, logging
import numpy as np
import nibabel as nib
from nibabel.trackvis import read as read_trk
from nibabel.trackvis import TrackvisFile
from ..utils import ExecCommand, DiciphrException
from ..nifti_utils import ( nifti_image, read_nifti, write_nifti, 
                threshold_image )
from ..diffusion import ( extract_b0, bet2_mask_nifti, 
                estimate_tensor, calculate_lmax_from_bvecs )
from dipy.io.streamline import load_trk, save_trk 
from dipy.tracking.life import transform_streamlines
from dipy.tracking import utils
from dipy.tracking.metrics import downsample
from dipy.tracking.streamline import Streamlines

##############################   
# COMMON PREPROCESSING STEPS #
##############################
def calculate_fa_threshold_mask(fa_im, threshold_value):
    return threshold_image(fa_im, threshold_value)
    
def get_lmax_from_bvecs(bvecs, lmax='auto'):
    max_possible_l = calculate_lmax_from_bvecs(bvecs)
    if lmax == 'auto':
        L = max_possible_l
        logging.info('Mode = {} lmax = {}'.format(lmax, L))
    elif lmax == 'minus_two':
        L = max_possible_l - 2 
        logging.info('Mode = {} lmax = {}'.format(lmax, L))
    else:
        try:
            L = int(lmax)
        except ValueError:
            raise DiciphrException('User provided non-integer argument for lmax not allowed.')
        if str(L).isdigit() and ( L % 2 == 0 ):
            if L > max_possible_l:
                raise DiciphrException('User provided lmax larger than allowed by unique bvecs in the gradient table.')
        else:
            raise DiciphrException('User provided forbidden value for lmax.')
        logging.info('Mode = {} lmax = {}'.format('user_input', L))
    return L 
    
def angle_threshold_to_curvature(angle_threshold, step_size):
    #http://www.nitrc.org/pipermail/mrtrix-discussion/2011-June/000230.html
    from math import sin
    angle_threshold_radians = (angle_threshold * 3.141592653589793) / 180.0
    curvature = float(step_size) / ( 2.0 * sin(angle_threshold_radians/2.0) )
    return curvature
    
def curvature_to_angle_threshold(curvature, step_size, radians=False):
    #http://www.nitrc.org/pipermail/mrtrix-discussion/2011-June/000230.html
    from math import asin
    angle_threshold = 2.0 * asin(step_size / (2.0 * curvature))
    if radians:
        return angle_threshold
    else:
        return 180.0 *(angle_threshold / np.pi)
    
###################################   
### COMMON POSTPROCESSING STEPS ###
###################################
def spline_filter(input_trk_file, output_trk_file, step_size=0.2):
    logging.debug('diciphr.tractography.track_utils.spline_filter')
    cmd = [ 'spline_filter', input_trk_file, step_size, output_trk_file]
    returncode, stdout, stderr = ExecCommand(cmd).run()
    return output_trk_file
    
def filter_tracks_include(input_trk_file, include_mask_file, output_trk_file):
    logging.debug('diciphr.tractography.track_utils.filter_tracks_include')
    include_mask_data = (nib.load(include_mask_file).get_data() > 0)
    logging.debug('nibabel.trackvis.read')
    trks, hdr = read_trk(input_trk_file, points_space='voxel')
    trks_round = [ (a[0].astype(np.int16), a[1], a[2]) for a in trks]
    trks_keep = []
    N = len(trks)
    logging.debug("Begin filter tracks include")
    for idx, tup in enumerate(trks_round):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, N))
        trk_arr = tup[0]
        for coord in trk_arr:
            if include_mask_data[coord[0],coord[1],coord[2]]:
                trks_keep.append(trks[idx])
                break
    logging.debug('nibabel.trackvis.TrackvisFile')
    hdr_keep= hdr.copy()
    hdr_keep['n_count'] = len(trks_keep) 
    T = TrackvisFile(trks_keep, hdr_keep, points_space='voxel')
    logging.debug('Write tracks to {}'.format(output_trk_file))
    T.to_file(output_trk_file)
    return output_trk_file
    
def filter_tracks_truncate(input_trk_file, truncate_mask_file, include_mask_file, output_trk_file):
    logging.debug('diciphr.tractography.track_utils.filter_tracks_truncate')
    include_mask_data = (nib.load(include_mask_file).get_data() > 0)
    truncate_mask_data = (nib.load(truncate_mask_file).get_data() > 0)
    logging.debug('nibabel.trackvis.read')
    trks, hdr = read_trk(input_trk_file, points_space='voxel')
    trks_keep = []
    _N = len(trks)
    streamlines = [a[0] for a in trks]
    _dtype = streamlines[0].dtype
    for idx, streamline in enumerate(streamlines):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, _N))
        M = len(streamline)
        streamline_keep=None
        streamline_rev_keep=None
        for coord in streamline:
            coord_int = coord.astype(np.int16)
            if streamline_keep is None:
                streamline_keep = coord[np.newaxis,:]
            else:
                streamline_keep = np.append(streamline_keep, coord[np.newaxis,:], axis=0)
            if truncate_mask_data[coord_int[0],coord_int[1],coord_int[2]]:
                break
        if len(streamline_keep) == len(streamline):
            # we never touched the stop mask
            trks_keep.append(streamline)
            continue
        for coord in streamline[::-1]:
            coord_int = coord.astype(np.int16)
            if streamline_rev_keep is None:
                streamline_rev_keep = coord[np.newaxis,:]
            else:
                streamline_rev_keep = np.append(streamline_rev_keep, coord[np.newaxis,:], axis=0)
            if truncate_mask_data[coord_int[0], coord_int[1], coord_int[2]]:
                break
        streamline_rev_keep = streamline_rev_keep[::-1]
        trks_keep.append(streamline_keep)
        trks_keep.append(streamline_rev_keep)
    trks = [(trk, None, None) for trk in trks_keep]
    #filter through include ROI
    trks_round = [ (a[0].astype(np.int16), a[1], a[2]) for a in trks]
    trks_keep = []
    _N = len(trks)
    for idx, tup in enumerate(trks_round):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, _N))
        trk_arr = tup[0]
        for coord in trk_arr:
            if include_mask_data[coord[0],coord[1],coord[2]]:
                trks_keep.append(trks[idx])
                break
    logging.debug('nibabel.trackvis.TrackvisFile')
    T = TrackvisFile(trks_keep, hdr, points_space='voxel')
    logging.debug('Write tracks to {}'.format(output_trk_file))
    T.to_file(output_trk_file)
    return output_trk_file
    
def filter_tracks_exclude(input_trk_file, exclude_mask_file, output_trk_file):
    logging.debug('diciphr.tractography.track_utils.filter_tracks_exclude')
    exclude_mask_data = (nib.load(exclude_mask_file).get_data() > 0)
    logging.debug('nibabel.trackvis.read')
    trks, hdr = read_trk(input_trk_file, points_space='voxel')
    trks_round = [ (a[0].astype(np.int16), a[1], a[2]) for a in trks]
    trks_keep = []
    N = len(trks)
    logging.debug("Begin filter tracks exclude")
    for idx, tup in enumerate(trks_round):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, N))
        trk_arr = tup[0]
        _keep = True
        for coord in trk_arr:
            if exclude_mask_data[coord[0],coord[1],coord[2]]:
                _keep = False
            if not _keep:
                break
        if _keep:
            trks_keep.append(trks[idx])
    logging.debug('nibabel.trackvis.TrackvisFile')
    hdr_keep= hdr.copy()
    hdr_keep['n_count'] = len(trks_keep) 
    T = TrackvisFile(trks_keep, hdr_keep, points_space='voxel')
    logging.debug('Write tracks to {}'.format(output_trk_file))
    T.to_file(output_trk_file)
    return output_trk_file
    
def track_fa_ratio(input_trk_file, fa_im):
    fa_data = fa_im.get_data()
    trks, hdr = read_trk(input_trk_file, points_space='voxel')
    trks_round = [ (a[0].astype(np.int16), a[1], a[2]) for a in trks]
    N = len(trks)
    global_min_fa = 1.0 
    fiber_mins_fa = np.ones((N,),dtype=float)
    for idx, tup in enumerate(trks_round):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, N))
        trk_arr = tup[0]
        fiber_voxel_data = np.zeros(fa_im.shape, dtype=int)
        for coord in trk_arr:
            fiber_voxel_data[coord[0],coord[1],coord[2]]=1
        fa_vec = fa_data[fiber_voxel_data > 0]
        fa_min = np.min(fa_vec)
        if fa_min < global_min_fa:
            global_min_fa = fa_min
        fiber_mins_fa[idx] = fa_min
    maxmin_fa = np.max(fiber_mins_fa)
    fa_ratio = maxmin_fa / global_min_fa
    logging.debug("Global Min FA: {}".format(global_min_fa))
    logging.debug("Max of min FA: {}".format(maxmin_fa))
    logging.debug("     FA ratio: {}".format(fa_ratio))
    return fa_ratio

def track_density_image(streamlines, ref_im):
    tdi = utils.density_map(streamlines, ref_im.affine, ref_im.shape)    
    tdi_im = nifti_image(np.asarray(tdi, dtype=np.uint8), ref_im.affine)
    return tdi_im
        
def downsample_tracks(streamlines, ref_im, downsample_percent=50, num_iters=15, return_mean_densities=False):
    '''
    Implementation of The fiber-density-coreset for redundancy reduction in huge fiber-sets
    Guy Alexandronia, Gali Zimmerman Morenob, Nir Sochenc, Hayit Greenspan
    Neuroimage
    DOI: 10.1016/j.neuroimage.2016.11.027
    '''
    N = len(streamlines)
    m = int(float(downsample_percent)/100*N)
    logging.info('The fiber-density-coreset for redundancy reduction in huge fiber-sets, Alexandronia et al.')
    logging.info('TDI image of streamlines')
    tdi_im = track_density_image(streamlines, ref_im)
    tdi_data = tdi_im.get_data().astype(float)
    streamlines_voxels = Streamlines(transform_streamlines(streamlines, np.linalg.inv(ref_im.affine)))
    # original paper subsampled each streamline to 20 coordinates. 
    # streamlines_downsampled = Streamlines([downsample(s, 20) for s in streamlines])
    visitation_map = (tdi_data > 0)
    nvoxels = float(np.sum(visitation_map))
    inverse_tdi_data = np.zeros(tdi_data.shape)
    inverse_tdi_data[visitation_map] = 1 / tdi_data[visitation_map]
    logging.info('Calculate mean inverse track density for each streamline')
    streamline_means = np.asarray([
            np.mean(np.take(inverse_tdi_data, np.ravel_multi_index(s.astype(int).T, tdi_data.shape))) 
            for s in streamlines_voxels
    ])
    # normalize to form pdf 
    streamline_means = streamline_means / np.sum(streamline_means)
    best_overlap = []
    def hamming_distance(sample_data):
        return np.sum(sample_data[visitation_map] > 0)/nvoxels
    best_h = 1
    best_streamlines = None
    logging.info('Randomly subsample {} streamlines {} times at {} % to downsample to {} streamlines'.format(
            N, num_iters, downsample_percent, m
    ))
    for iter in range(num_iters):
        logging.info('Iteration {} out of {}'.format(iter+1, num_iters))
        c = Streamlines(np.random.choice(streamlines, (m,), p=streamline_means))
        sample_data = track_density_image(c, ref_im).get_data() 
        h = hamming_distance(sample_data)
        logging.debug('Hamming distance {} best {}'.format(h, best_h))
        if h <= best_h:
            best_streamlines = c
            best_h = h
    if return_mean_densities:
        return best_streamlines, streamline_means
    else:
        return best_streamlines         
              
### CAMINO : QUTE , RBF etc. ###
#  This class provides an interface to read and manipulate camino fibers
class caminoFibers:
    ## @brief Constructor
    #
    #  Either initialize an empty fiber array or read fiber coordinates from a file/array
    #
    #  @param fiberFile is the path for .Bfloat file. Read fibre coordinates from the file if provided
    #  @param F is an array with pre-read coordinates. Directly copy coordinates form the array if provided
    def __init__(self, fiberFile="", F=None):
        if fiberFile != "":
            self.readFibersFromFile(fiberFile)
        elif F != None:
            self.F = F[:]
            self.numFibers = len(F)
        else:
            self.F = list()
            self.numFibers = 0

    ## @brief Read fiber coordinates from the provided file
    #
    #  @param fiberFile is the path for .Bfloat file.
    def readFibersFromFile(self, fiberFile):
        import struct

        self.F = list()

        f = open(fiberFile, "rb")
        while 1:
            # read number of points for the tract
            byte = f.read(4)
            if byte == '' :
                break
            numPoints = int(struct.unpack('>f', byte)[0])
            byte = f.read(4)

            # read the seedpoint
            seedPoint = int(struct.unpack('>f', byte)[0])

            #read physical coordinates
            coords = np.zeros([numPoints, 3])
            for i in range(0, numPoints):
                byte = f.read(4)
                x = float(struct.unpack('>f', byte)[0])
                byte = f.read(4)
                y = float(struct.unpack('>f', byte)[0])
                byte = f.read(4)
                z = float(struct.unpack('>f', byte)[0])

                coords[i, :] = [x,y,z]
            self.F.append(coords)

        self.numFibers = len(self.F)
        f.close()

    ## @brief Generate connectivity signatures for fibers
    #
    #  Coordinate system for fibers and images of streamline counts muct be same. User must be sure of this since
    #  there is no way to get what coordinate system was used for fibers from .Bfloat file.
    #
    #  @param fiberFile is the path for .Bfloat file.
    #  @param Images is the array including images of streamline counts. Each image corresponds to a single target region.
    #  @param header is the nifti image to extract coordinate system of the fibers
    #  @param outputFile is the csv file for the output
    def generateConnectivitySig(self, fiberFile, Images, header, outputFile):
        from scipy import ndimage
        #import nifti as nif
        from nibabel import load as load_nifti
        import csv
        import struct

        numGM = len(Images)

        featureFile = open(outputFile, "w")
        writer = csv.writer(featureFile, delimiter=',')

        # mappings for coordinate transformation
        #niftiImage = nif.NiftiImage(header)
        #mV2P = niftiImage.header['sform'][0:3, 0:3]
        #mP2V = np.linalg.inv(mV2P)
        #origin = niftiImage.header['sform'][0:3, 3].reshape([3, 1])

        niftiImage = load_nifti(header)
        mV2P = niftiImage.affine[0:3,0:3]
        mP2V = np.linalg.inv(mV2P)
        origin = niftiImage.affine[0:3, 3].reshape([3, 1])

        f = open(fiberFile, "rb")
        #counter = 0
        while 1:
            # read number of points for the tract
            byte = f.read(4)
            if byte == '' :
                break
            numPoints = int(struct.unpack('>f', byte)[0])
            byte = f.read(4)

            # read the seedpoint
            seedPoint = int(struct.unpack('>f', byte)[0])

            #read physical coordinates
            fiberXYZ = np.zeros([numPoints, 3])
            for i in range(0, numPoints):
                byte = f.read(4)
                x = float(struct.unpack('>f', byte)[0])
                byte = f.read(4)
                y = float(struct.unpack('>f', byte)[0])
                byte = f.read(4)
                z = float(struct.unpack('>f', byte)[0])

                fiberXYZ[i, :] = [x,y,z]

            #convert to voxel space
            fiberIJK = np.ceil(np.dot(mP2V, fiberXYZ.T - np.tile(origin, [1, numPoints])).T).astype(int)

            #get unique coordinates
            _fiberIJK = [fiberIJK[0, :]]
            for vox in range(1, fiberIJK.shape[0]):
                if np.sum(_fiberIJK[-1]==fiberIJK[vox, :]) < 3:
                    _fiberIJK.append(fiberIJK[vox, :])
            fiberIJK = np.array(_fiberIJK)

            #fill frequencies from Images
            numVox = fiberIJK.shape[0]
            Pf = np.zeros((numVox, numGM))
            for v in range(0, numVox):
                for r in range(0, numGM):
                    #beware that nifti package switches x and z coordinates
                    #nibabel package does not switch x and z -- drew
                    corX = min(fiberIJK[v,0], Images[r].shape[0]-1)  #min(fiberIJK[v,2],
                    corY = min(fiberIJK[v,1], Images[r].shape[1]-1)
                    corZ = min(fiberIJK[v,2], Images[r].shape[2]-1)  #min(fiberIJK[v,0],
                    Pf[v, r] = Images[r][corX,corY,corZ]

            #apply smoothing
            Lf = ndimage.gaussian_filter(Pf, sigma=[5,0])

            #weighted average of voxels
            arclength = Lf.shape[0]
            ss = np.arange(0,arclength) / float(arclength-1. + 1e-16)
            wmin = 0.01
            ss[0] = 1e-2; ss[arclength-1] = 1.-(1e-2)
            ww0 = (ss**(-0.5))*((1-ss)**(-0.5)) + wmin
            ww = ww0 / np.sum(ww0)
            F = np.sum(Lf*ww.reshape((len(ww),1)), axis=0)

            #convert to integer and store
            fV = F.astype('int')
            writer.writerow(fV)

            #counter += 1
            #print("%d fibers were processed"%(counter))

        f.close()
        featureFile.close()


## @brief Extracts selected tratcs from a set of fibers
#
#  @param bfloatFile is the path for .Bfloat file.
#  @param clusterIDs is the list including cluster ids of fibers.
#  @param selectedTracts is the list of selected cluster ids for each tract.
#  @param outputFileNames is the list of names output tract files.
def extractTracts(bfloatFile, clusterIDs, selectedTracts, outputFileNames):
    import struct

    #output files
    tractFiles = []
    for fileName in outputFileNames:
        _f = open(fileName, "wb")
        tractFiles.append(_f)

    #read the .Bfloat file and write the fibers that are selected
    f = open(bfloatFile, "rb")
    for fib in range(len(clusterIDs)):
        # read number of points for the tract
        byte = f.read(4)
        if byte == '' :
            print("File %s ends unexpectedly" % (bfloatFile))
            break
        numPoints = int(struct.unpack('>f', byte)[0])
        byte = f.read(4)

        # read the seedpoint
        seedPoint = int(struct.unpack('>f', byte)[0])

        #read physical coordinates
        coords = np.zeros([numPoints, 3])
        for i in range(0, numPoints):
            byte = f.read(4)
            x = float(struct.unpack('>f', byte)[0])
            byte = f.read(4)
            y = float(struct.unpack('>f', byte)[0])
            byte = f.read(4)
            z = float(struct.unpack('>f', byte)[0])
            coords[i, :] = [x,y,z]

        #check if we want this fiber and then write it to the right file
        for tr in range(len(selectedTracts)):
            if clusterIDs[fib] in selectedTracts[tr]:
                _f = tractFiles[tr]

                s = struct.pack('>f', numPoints)
                _f.write(s)
                s = struct.pack('>f', seedPoint)
                _f.write(s)
                for p in range(0, numPoints):
                    for x in range(0, 3):
                        s = struct.pack('>f', coords[p, x])
                        _f.write(s)
                break

    #close input and output files
    f.close()
    for i in range(len(outputFileNames)):
        tractFiles[i].close()


def convertP2V(fiberXYZ, mP2V, origin, imageSize):
    fiberIJK = np.floor(np.dot(mP2V, fiberXYZ.T - origin).T).astype(int)
    fiberIJK[fiberIJK[:,0]>=imageSize[0], 0] = imageSize[0]-1
    fiberIJK[fiberIJK[:,1]>=imageSize[1], 1] = imageSize[1]-1
    fiberIJK[fiberIJK[:,2]>=imageSize[2], 2] = imageSize[2]-1
    return fiberIJK

def convertV2P(fiberIJK, mV2P, origin, imageSize):
    fiberXYZ = (np.dot(mV2P, fiberIJK.T) + origin).T
    return fiberXYZ
