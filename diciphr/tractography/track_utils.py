import os, logging
import numpy as np
from diciphr.utils import ExecCommand, DiciphrException
from diciphr.nifti_utils import read_nifti, write_nifti, nifti_image, threshold_image
from diciphr.diffusion import calculate_lmax_from_bvecs
from dipy.io.streamline import load_trk, save_trk 
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import density_map, target
from dipy.tracking.streamline import length, values_from_volume
import xml.etree.ElementTree as ET

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
    angle_threshold_radians = (angle_threshold * np.pi) / 180.0
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
def track_info(trk_input):
    if isinstance(trk_input, StatefulTractogram):
        sft = trk_input
    else:
        sft = load_trk(trk_input, reference='same')
    return {
        'number streamlines': len(sft), 
        'affine': sft.space_attributes[0], 
        'dimensions': sft.space_attributes[1], 
        'voxel sizes': sft.space_attributes[2], 
        'orientation': sft.space_attributes[3], 
    }
    
def track_stats(trk_input, nifti_scalars={}):
    if isinstance(trk_input, StatefulTractogram):
        sft = trk_input
    else:
        sft = load_trk(trk_input, reference='same')
    streamline_lengths = length(sft.streamlines)
    tdi_img = track_density_image(sft)
    nvoxels = np.sum(tdi_img.get_fdata() > 0)
    zooms = tdi_img.header.get_zooms()
    volume = nvoxels * zooms[0] * zooms[1] * zooms[2] / 1000.0
    # to do mean FA of streamlines or of voxels 
    ret_dict = {
        'number streamlines': len(sft), 
        'min length': np.min(streamline_lengths),
        'max length': np.max(streamline_lengths),
        'mean length': np.mean(streamline_lengths),
        'SD length' : np.std(streamline_lengths), 
        'number voxels': nvoxels, 
        'volume' : volume
    }
    if nifti_scalars:
        sft = StatefulTractogram.from_sft(sft.streamlines, sft)
        sft.to_rasmm()
    for scalar_name, scalar_img in nifti_scalars.items():
        scalar_values = values_from_volume(scalar_img.get_fdata(), sft.streamlines, affine=scalar_img.affine)
        scalar_values = np.concatenate(scalar_values)
        scalar_values = scalar_values[np.isfinite(scalar_values)]
        ret_dict['min '+scalar_name] = np.min(scalar_values)
        ret_dict['max '+scalar_name] = np.max(scalar_values)
        ret_dict['mean '+scalar_name] = np.mean(scalar_values)
        ret_dict['SD '+scalar_name] = np.std(scalar_values)
    return ret_dict
        
    
def spline_filter(input_trk_file, output_trk_file, step_size=0.2):
    logging.debug('diciphr.tractography.track_utils.spline_filter')
    cmd = ['spline_filter', input_trk_file, step_size, output_trk_file]
    ExecCommand(cmd).run()
    return output_trk_file
    
def filter_tracks_include(input_trk_file, output_trk_file, *include_mask_files):
    """
    Filters streamlines from a .trk file that pass through a binary inclusion mask using DIPY's `target`.

    Parameters:
    - input_trk_file: str, path to input .trk file
    - include_mask_file: str, path to binary inclusion mask (.nii or .nii.gz)
    - output_trk_file: str, path to save the filtered .trk file
    """
    logging.debug('Loading inclusion masks')
    include_masks = []
    for incl_nii in include_mask_files:
        img = read_nifti(incl_nii)
        include_masks.append(img.get_fdata()>0)        
    logging.debug('Loading tractogram')
    sft = load_trk(input_trk_file, reference='same')
    filtered_streamlines = sft.streamlines 
    for include_mask in include_masks:
        logging.debug('Filtering streamlines using target with include=True')
        print(sft.affine)
        filtered_streamlines = target(filtered_streamlines, sft.affine, include_mask, include=True)
    filtered_sft = StatefulTractogram.from_sft(
        filtered_streamlines,
        sft
    )
    save_trk(filtered_sft, output_trk_file)
    logging.debug(f'Filtered tractogram saved to {output_trk_file}')
    return output_trk_file
    
def filter_tracks_exclude(input_trk_file, output_trk_file, *exclude_mask_files):
    """
    Filters streamlines from a .trk file that do not pass through a binary inclusion mask using DIPY's `target`.

    Parameters:
    - input_trk_file: str, path to input .trk file
    - include_mask_file: str, path to binary inclusion mask (.nii or .nii.gz)
    - output_trk_file: str, path to save the filtered .trk file
    """
    logging.debug('Loading inclusion masks')
    exclude_masks = []
    for excl_nii in exclude_mask_files:
        img = read_nifti(excl_nii)
        exclude_masks.append(img.get_fdata()>0)        
    logging.debug('Loading tractogram')
    sft = load_trk(input_trk_file, reference='same')
    filtered_streamlines = sft.streamlines 
    for exclude_mask in exclude_masks:
        logging.debug('Filtering streamlines using target with include=True')
        print(sft.affine)
        filtered_streamlines = target(filtered_streamlines, sft.affine, exclude_mask, include=False)

    filtered_sft = StatefulTractogram.from_sft(
        filtered_streamlines,
        sft
    )
    save_trk(filtered_sft, output_trk_file)
    logging.debug(f'Filtered tractogram saved to {output_trk_file}')
    return output_trk_file

def track_density_image(trk_input, ref_img=None, output_filename=None):
    if isinstance(trk_input, StatefulTractogram):
        sft = trk_input
    else:
        sft = load_trk(trk_input, reference=ref_img if ref_img else 'same')
    sft_copy = StatefulTractogram.from_sft(sft.streamlines, sft, data_per_point=sft.data_per_point, data_per_streamline=sft.data_per_streamline)
    sft_copy.to_rasmm()
    if ref_img is None:
        ref_affine = sft.space_attributes[0]
        ref_shape = sft.space_attributes[1]
    else:
        ref_affine = ref_img.affine
        ref_shape = ref_img.shape
    tdi = density_map(sft_copy.streamlines, ref_affine, ref_shape)    
    tdi_im = nifti_image(np.asarray(tdi, dtype=np.uint8), ref_affine)
    if output_filename is not None:
        write_nifti(output_filename, tdi_im)
    return tdi_im

def downsample_tracks_fdc(trk_input, output_trk_filename=None, ref_filename=None, downsample_percent=50, num_iters=15, return_mean_densities=False):
    '''
    Downsamples streamlines from a .trk file using the fiber-density-coreset method.
    Based on: Alexandronia et al., Neuroimage, DOI: 10.1016/j.neuroimage.2016.11.027

    Parameters:
    - trk_input: path to the .trk file or StatefulTractogram instance
    - ref_filename: optional path to a NIfTI file to use as reference image
    - downsample_percent: percentage of streamlines to retain
    - num_iters: number of random subsampling iterations
    - return_mean_densities: whether to return streamline density weights
    '''
    if ref_filename is not None:
        logging.info(f'Reference nifti: {ref_filename}')
        ref_img = read_nifti(ref_filename)
    else:
        ref_img = None
    if isinstance(trk_input, StatefulTractogram):
        sft = trk_input
    else:
        logging.info(f'Load trk file {trk_input}')
        sft = load_trk(trk_input, reference=ref_img if ref_img else 'same')     
    
    #sft.to_rasmm()
    N = len(sft.streamlines)
    m = int(float(downsample_percent) / 100 * N)

    logging.info('Computing TDI image')
    tdi_data = track_density_image(sft, ref_img=ref_img).get_fdata()

    visitation_map = tdi_data > 0
    nvoxels = float(np.sum(visitation_map))

    inverse_tdi_data = np.zeros_like(tdi_data)
    inverse_tdi_data[visitation_map] = 1 / tdi_data[visitation_map]

    logging.info('Calculating mean inverse track density for each streamline')
    sft_vox_copy = StatefulTractogram.from_sft(sft.streamlines, sft)
    sft_vox_copy.to_vox()
    sft_vox_copy.to_center()
    streamline_means = np.asarray([
        np.mean(np.take(inverse_tdi_data, np.ravel_multi_index(s.astype(np.int32).T, tdi_data.shape)))
        for s in sft_vox_copy.streamlines
    ])
    streamline_means /= np.sum(streamline_means)

    best_h = 1
    best_sft = None
    logging.info(f'Subsampling {N} streamlines {num_iters} times at {downsample_percent}% to get {m} streamlines')
    for iter in range(num_iters):
        logging.debug(f'Iteration {iter + 1} of {num_iters}')
        sampled_indices = np.random.choice(np.arange(N), size=m, replace=False, p=streamline_means)
        sampled_sft = sft[sampled_indices]
        sample_data = track_density_image(sampled_sft, ref_img=ref_img).get_fdata()
        # Hamming distance 
        h = np.sum(sample_data[visitation_map] > 0) / nvoxels
        logging.debug(f'Hamming distance {h}, best so far {best_h}')
        if h <= best_h:
            best_sft = sampled_sft
            best_h = h
    
    # save results
    if output_trk_filename is not None:
        logging.info(f"Save results to {output_trk_filename}")
        save_trk(best_sft, output_trk_filename)
    
    if return_mean_densities:
        return best_sft, streamline_means
    else:
        return best_sft

###################################   
###  TRACKVIS MERGE AND SCENE   ###
###################################
def merge_tracks(input_trks, output_trk_file=None):
    """
    Merges multiple .trk files into one, assigning a 'DataSetID' to each streamline
    based on the order of input files.

    Parameters:
    - input_trk_files: list of str, paths to input .trk files
    - output_trk_file: str, path to save the merged .trk file
    """
    all_streamlines = []
    all_dataset_ids = [] 
    ref_sft = None
    for dataset_id, trk_input in enumerate(input_trks):
        if isinstance(trk_input, StatefulTractogram):
            sft = trk_input
        else:
            sft = load_trk(trk_input, reference='same')
        if ref_sft is None:
            ref_sft = sft 
        streamlines = list(sft.streamlines)
        data_per_streamline = [dataset_id] * len(streamlines)
        all_streamlines.extend(streamlines)
        all_dataset_ids.extend(data_per_streamline)
    data_per_streamline = {"DataSetID": np.asarray(all_dataset_ids, dtype=np.float32)[:,np.newaxis]}
    all_streamlines  = np.asarray(all_streamlines)
    merged_sft = StatefulTractogram.from_sft(
            all_streamlines, 
            ref_sft,
            data_per_streamline = data_per_streamline
    )
    if output_trk_file:
        save_trk(merged_sft, output_trk_file)
    return merged_sft    

_default_colors = [
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (0,255,255),
    (255,0,255),
    (255,255,0),
    (255,165,0),
    (191,255,0),
    (64,224,208),
    (255,105,180),
    (138,43,226),
    (127,255,0),
    (0,191,255),
    (220,20,60),
    (0,255,127),
    (255,215,0),
]

def create_scene(trk_path, output_path, underlay_images=[], roi_images=[], roi_names=[], roi_colors=[], roi_opacities=[],
                 track_group_names=[], track_colors=[], solid_colors=False, render="Line"):
    if not track_colors:
        track_colors = _default_colors
    if not roi_colors:
        roi_colors = _default_colors
    trk = load_trk(trk_path, reference='same')
    data_per_streamline = trk.data_per_streamline
    affine, dims, pixdims, orn = trk.space_attributes
    half_dims = [str(int(d/2)) for d in dims]
    scene = ET.Element("Scene", version="2.2")
    ET.SubElement(scene, "Dimension", x=str(dims[0]), y=str(dims[1]), z=str(dims[2]))
    ET.SubElement(scene, "VoxelSize", x=str(pixdims[0]), y=str(pixdims[1]), z=str(pixdims[2]))
    ET.SubElement(scene, "VoxelOrder", current=orn, original=orn)
    ET.SubElement(scene, "LengthUnit", value="1")
    ET.SubElement(scene, "TrackFile", path=trk_path, rpath=os.path.basename(trk_path))

    if underlay_images:
        image = ET.SubElement(scene, "Image")
        ET.SubElement(image, "SliceX", number=half_dims[0], visibility="1")
        ET.SubElement(image, "SliceY", number=half_dims[1], visibility="1")
        ET.SubElement(image, "SliceZ", number=half_dims[2], visibility="1")
        ET.SubElement(image, "Interpolate", value="0")
        ET.SubElement(image, "Opacity", value="0.5")
        for underlay_image in underlay_images:
            _dat = read_nifti(underlay_image).get_fdata()
            _perc99 = np.percentile(_dat[_dat!=0], 99)
            ET.SubElement(image, "Map", path=underlay_image, rpath=os.path.basename(underlay_image),
                      rpath2=os.path.basename(underlay_image), window=f"{_perc99:0.6f}", level=f"{(_perc99/2):0.6f}")
        ET.SubElement(image, "CurrentIndex", value="0")

    if roi_images:
        rois = ET.SubElement(scene, "ROIs")
        for i, roi_path in enumerate(roi_images):
            try:
                roiname = roi_names[i]
            except IndexError:
                roiname = os.path.basename(roi_path)
            try:
                opacity = str(float(roi_opacities[i]))
            except IndexError:
                opacity = str(float(1))
            roi = ET.SubElement(rois, "ROI", name=roiname, type="FromImage", id=str(1000+i))
            ET.SubElement(roi, "ImageFile", path=roi_path, rpath=os.path.basename(roi_path),
                          rpath2=os.path.basename(roi_path), low="0.25", high="1")
            ET.SubElement(roi, "Edited", value="0")
            color1, color2, color3 = roi_colors[i%len(roi_colors)]
            ET.SubElement(roi, "Color", r=str(color1), g=str(color2), b=str(color3))
            ET.SubElement(roi, "Opacity", value=opacity)
            ET.SubElement(roi, "Visibility", value="1")
        ET.SubElement(rois, "CurrentIndex", value="0")

    tracks = ET.SubElement(scene, "Tracks")
    
    render_dict = {
        "lin":"0",
        "tub":"1",
        "sha":"2",
        "end":"3"            
            }
    
    if data_per_streamline:
        unique_props = list(np.unique(data_per_streamline[list(data_per_streamline.keys())[0]]))
    else:
        unique_props = [0.0]
    for prop in unique_props:
        i = int(prop)
        try:
            trackname = track_group_names[i]
        except IndexError:
            trackname = f"Track {i+1:03d}"
        track = ET.SubElement(tracks, "Track", name=trackname, id=str(2000+i))
        ET.SubElement(track, "Length", low="0", high="1e+08")
        ET.SubElement(track, "FileIDs", value="0")
        ET.SubElement(track, "Property", id="0", low=str(prop), high=str(prop))
        ET.SubElement(track, "Slice", plane="0", number=half_dims[0], thickness="1", testmode="0", enable="0", visible="1", operator="and", id=str(20000+3*i))
        ET.SubElement(track, "Slice", plane="1", number=half_dims[1], thickness="1", testmode="0", enable="0", visible="1", operator="and", id=str(20000+3*i+1))
        ET.SubElement(track, "Slice", plane="2", number=half_dims[2], thickness="1", testmode="0", enable="0", visible="1", operator="and", id=str(20000+3*i+2))
        ET.SubElement(track, "Skip", value="0", enable="0")
        ET.SubElement(track, "ShadingMode", value=render_dict[render.lower()[:3]])
        ET.SubElement(track, "Radius", value="0.05")
        ET.SubElement(track, "NumberOfSides", value="5")
        ET.SubElement(track, "ColorCode", value="1" if solid_colors else "0")
        ET.SubElement(track, "OdfColorCode", value="0")
        color1, color2, color3 = track_colors[i%len(track_colors)]
        ET.SubElement(track, "SolidColor", r=str(color1), g=str(color2), b=str(color3))
        ET.SubElement(track, "ScalarIndex", value="0")
        gradient = ET.SubElement(track, "ScalarGradient")
        ET.SubElement(gradient, "ColorStop", stop="0", r="1", g="1", b="0")
        ET.SubElement(gradient, "ColorStop", stop="1", r="1", g="0", b="0")
        ET.SubElement(track, "Saturation", value="1")
        ET.SubElement(track, "HelixPoint", x=half_dims[0], y=half_dims[1], z=half_dims[2])
        ET.SubElement(track, "HelixVector", x="1", y="0", z="0")
        ET.SubElement(track, "HelixAxis", visibility="1")
        ET.SubElement(track, "Visibility", value="1")
    ET.SubElement(tracks, "CurrentIndex", value="0")

    tree = ET.ElementTree(scene)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Scene file created at {output_path}")
