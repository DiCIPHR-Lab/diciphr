import logging
import numpy as np
import nibabel as nib
from nibabel.trackvis import read as read_trk
from nibabel.trackvis import TrackvisFile

def track_density_image(input_trk_file, ref_im):
    '''
    Calculates a track density image from a trk tractograpy file and a reference image for header geometry information. 
    
    Parameters
    ----------
    input_trk_file : str
        Filename of the tractography file. 
    ref_im : nibabel.Nifti1Image
        Reference image, such as FA or brain mask.
        
    Returns
    -------
    nibabel.Nifti1Image
        Tract Density Image 
    '''
    logging.debug('diciphr.tractography.track_density_image')
    logging.debug('nibabel.trackvis.read')
    trks, hdr = read_trk(input_trk_file, points_space='voxel')
    trks_round = [ (a[0].astype(np.int16), a[1], a[2]) for a in trks]
    tdi_data = np.zeros(ref_im.shape,dtype=np.int32)
    affine = ref_im.affine
    _N = len(trks)
    logging.debug("Begin track density image")
    for idx, tup in enumerate(trks_round):
        logging.debug("Inspecting track {0} out of {1}".format(idx+1, _N))
        trk_arr = tup[0]
        fiber_voxel_data = np.zeros(ref_im.shape, dtype=int)
        for coord in trk_arr:
            fiber_voxel_data[coord[0],coord[1],coord[2]]+=1
        fiber_voxel_data[fiber_voxel_data > 0] = 1
        tdi_data += fiber_voxel_data
    tdi_im = nib.Nifti1Image(tdi_data, affine)
    tdi_im.header.set_qform(affine)
    tdi_im.header.set_sform(affine)
    tdi_im.update_header()
    return tdi_im
    
def cohens_kappa_tdi(tdi_im1, tdi_im2, brain_mask_im):
    '''
    Calculates cohen's kappa with two Track Density Images
    
    Parameters
    ----------
    tdi_im1 : nibabel.Nifti1Image
        First track density image.
    tdi_im2 : nibabel.Nifti1Image
        Second track density image.
    brain_mask_im : nibabel.Nifti1Image
        Brain mask image. Use all voxels in brain. 
    
    Returns
    -------
    float
        Cohen's kappa value
    '''
    tdi_data1=(tdi_im1.get_data() > 0).astype(int)  # 0 and 1 
    tdi_data2=(tdi_im2.get_data() > 0).astype(int)  # 0 and 1 
    tdi_intersection_data = tdi_data1 * tdi_data2
    total_voxels=np.sum(brain_mask_im.get_data() > 0)  #universal set 
    num_voxels1=np.sum(tdi_data1)   #A
    num_voxels2=np.sum(tdi_data2)   #B 
    pp = np.sum(tdi_intersection_data)
    pn = num_voxels1 - pp  # 1 in tdi_im1, 0 in tdi_im2
    np = num_voxels2 - pp  # 0 in tdi_im1, 1 in tdi_im2 
    nn = total_voxels - pn - np - pp 
    Enn = ((np + nn) * (pn + nn)) / float(total_voxels)
    Epp = ((np + pp) * (pn + pp)) / float(total_voxels)
    Oa = 1. * (nn + pp) / total_voxels
    Ea = 1. * (Enn + Epp) / total_voxels
    kappa = (Oa - Ea) / (1. - Ea)
    return kappa
    
