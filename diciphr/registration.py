#! /usr/bin/env python

import os, logging, shutil
from diciphr.utils import DiciphrException, ExecCommand, TempDirManager, force_to_list, which
from diciphr.nifti_utils import ( write_nifti, resample_image, apply_mask_to_image, threshold_image )
import numpy as np
    
def ants_registration_dti_t1(output_prefix, b0_img, t1_img, fa_img=None, t1_mask_img=None, dti_mask_img=None,
        resample_inputs=True, initial=None, syn=False, phase_enc='AP'):
    weight_fa=0.3
    weight_b0=0.7 
    if syn:
        if phase_enc.upper() in ['LR','RL']:
            restrict_deformation = '1.0x0.1x0.1'
        if phase_enc.upper() in ['AP','PA']:
            restrict_deformation = '0.1x1.0x0.1'
        if phase_enc.upper() in ['SI','IS']:
            restrict_deformation = '0.1x0.1x1.0'
    
    if t1_mask_img is None:
        t1_mask_img = threshold_image(t1_img)
    if dti_mask_img is None:
        dti_mask_img = threshold_image(b0_img)
        
    # keep original resolution images as reference to write outputs 
    dti_ref_img = b0_img 
    t1_ref_img = t1_img 
    if resample_inputs:
        t1_img = resample_image(t1_img, [1,1,1]) 
        b0_img = resample_image(b0_img, [1,1,1]) 
        t1_mask_img =  resample_image(t1_mask_img, master=t1_img, interp='NearestNeighbor')
        dti_mask_img = resample_image(dti_mask_img, master=b0_img, interp='NearestNeighbor')
        if fa_img:
            fa_img = resample_image(fa_img, master=b0_img)
    
    # Apply masks to data 
    t1_masked_img = apply_mask_to_image(t1_img, t1_mask_img)
    b0_masked_img = apply_mask_to_image(b0_img, dti_mask_img)
    fa_masked_img = None 
    if fa_img:
        fa_masked_img = apply_mask_to_image(fa_img, dti_mask_img)
    
    with TempDirManager(prefix='Registration_DTI-T1') as manager:
        tmpdir = manager.path()
        tmp1prefix = os.path.join(tmpdir, 'rigid_dti-t1') 
        tmp2prefix = os.path.join(tmpdir, 'syn_dti-t1') 
        b0filename = os.path.join(tmpdir, 'b0.nii.gz')
        t1filename = os.path.join(tmpdir, 't1.nii.gz')
        fafilename = os.path.join(tmpdir, 'fa.nii.gz')
        b0_target = os.path.join(tmpdir, 'b0_dico.nii.gz')
        t1ref_filename = os.path.join(tmpdir, 't1_ref_img.nii.gz')
        dtiref_filename  = os.path.join(tmpdir, 'dti_ref_img.nii.gz')
        t1rigid_filename = os.path.join(tmpdir, 'T1-DTI.nii.gz')
        write_nifti(b0filename, b0_masked_img)
        write_nifti(t1filename, t1_masked_img)
        write_nifti(t1ref_filename, t1_ref_img)
        write_nifti(dtiref_filename, dti_ref_img)
        if fa_img:
            write_nifti(fafilename, fa_masked_img)
        
        # antsAI
        logging.info("Auto initialization with antsAI")
        antsAI_transform = os.path.join(tmpdir, 'antsAI_rigid.txt')
        antsAI(t1filename, b0filename, antsAI_transform)
        # rigid 
        logging.info("Rigidly register DTI to T1")
        AR = AntsRegistration(tmp1prefix, initialize=antsAI_transform, winsorize=[0.025,0.975])
        AR.add_stage([t1filename], [b0filename], 'Rigid', 'MI', [100,70,50,20], [8,4,2,1], [3,2,1,0],
                         metric_weights=[1],
                         samplingStrategy='Regular', samplingPercentage=0.25, smoothing_unit='vox')
        AR.run()
        dtit1rigid = f"{tmp1prefix}0GenericAffine.mat"
        if syn:
            AR = AntsRegistration(tmp2prefix, initialize=dtit1rigid, invert_initial=True, initial_xfm_moving=False, winsorize=[0.025,0.975])
            if fa_img:
                AR.add_stage([t1filename, t1filename], [b0filename, fafilename], 'SyN', 'MI', [100,70,50,20], [8,4,2,1], [3,2,1,0],
                         metric_weights=[weight_b0, weight_fa], restrict_deformation=restrict_deformation) #
                         #samplingStrategy='Regular', samplingPercentage=0.25, smoothing_unit='vox')
            else:
                AR.add_stage([t1filename], [b0filename], 'SyN', 'MI', [100,70,50,20], [8,4,2,1], [3,2,1,0],
                         metric_weights=[1], restrict_deformation=restrict_deformation) #,
                         #convergence_threshold=1e-6, convergence_window=10,
                         #samplingStrategy='Regular', samplingPercentage=0.25, smoothing_unit='vox')
            AR.run()
            dicowarp = f"{tmp2prefix}Warp.nii.gz"
            dicoinvwarp = f"{tmp2prefix}InverseWarp.nii.gz"
            # Apply dico warp to B0 image 
            ants_apply_transforms(b0filename, b0_target, b0filename, [dicowarp])    
            # Redo rigid with dico warped B0 image 
            AR = AntsRegistration(tmp1prefix, initialize=antsAI_transform, winsorize=[0.025,0.975])
            AR.add_stage([t1filename], [b0_target], 'Rigid', 'MI', [100,70,50,20], [8,4,2,1], [3,2,1,0],
                             metric_weights=[1]) #,
                             # samplingStrategy='Regular', samplingPercentage=0.25, smoothing_unit='vox')
            AR.run()
            ants_apply_transforms(t1ref_filename, t1rigid_filename, dtiref_filename, [dicowarp, dtit1rigid], invert=[0,1])  
            shutil.copyfile(dicowarp, f"{output_prefix}_0dicoWarp.nii.gz")
            shutil.copyfile(dicoinvwarp, f"{output_prefix}_0dicoInverseWarp.nii.gz")
        else:
            ants_apply_transforms(t1ref_filename, t1rigid_filename, dtiref_filename, [dtit1rigid], invert=[1])  
        shutil.copyfile(dtit1rigid, f"{output_prefix}_DTI-T1-0GenericAffine.mat")
        shutil.copyfile(t1rigid_filename, f"{output_prefix}_T1-DTI.nii.gz")
    
#def ants_registration_t1_eve():
#def ants_registration_dti_eve():
#def ants_registration_t1_mni():
#def ants_registration_intrasubject(fixed_filename, moving_filename, outdir, subject, 
#                transform='r', initialize='identity', ):


def antsAI(fixed_filename, moving_filename, output_transform_file, dimensionality=3, verbose=True, transform='Rigid'):
    cmd  = ['antsAI', '-d', str(dimensionality), '-t', transform, \
                '-m', f"Mattes[{fixed_filename},{moving_filename},32,Regular,0.25]", 
                '-o', output_transform_file, '-p', '0', '-v', (1 if verbose else 0)]
    return ExecCommand(cmd).run()

class AntsRegistration():
    def __init__(self, output_prefix, initialize='identity', initial_fixed=None, initial_moving=None, 
                invert_initial=False, initial_xfm_moving=True, dimensionality=3, warped_outputs=2, interpolation='Linear', 
                histogram_matching=False, winsorize=[0.005,0.995], float=False, verbose=True
        ):    
        self.cmd=['antsRegistration']
        self.stages = []
        
        self.set_global_options(dimensionality=dimensionality, verbose=verbose, float=float, 
            interpolation=interpolation, histogram_matching=histogram_matching, winsorize=winsorize)
        if os.path.exists(initialize):
            self.set_initial(initialize, initial_xfm_moving=initial_xfm_moving, invert=invert_initial)
        elif initial_fixed and initial_moving:
            self.set_initial(initialize, initial_fixed, initial_moving, invert_initial)
        self.set_outputs(output_prefix, warped_outputs)
    
    def set_outputs(self, output_prefix, warped_outputs=2):
        # set up outputs 
        warped = output_prefix+'Warped.nii.gz'
        invwarped = output_prefix+'InverseWarped.nii.gz'
        self.outputs = []
        if warped_outputs == 0:
            self.outputs.extend(['--output',output_prefix])
        elif warped_outputs == 1:
            self.outputs.extend(['--output',f'[{output_prefix},{warped}]'])
        elif warped_outputs == 2:
            self.outputs.extend(['--output',f'[{output_prefix},{warped},{invwarped}]'])
        else:
            raise DiciphrException("Invalid valie of warped_outputs")
        
    def set_global_options(self, dimensionality=3, verbose=True, float=False, interpolation='Linear', 
                            histogram_matching=False, winsorize=[0.005,0.995]):
        self.global_options = ['--dimensionality',str(dimensionality)]
        self.global_options.extend(['--verbose',('1' if verbose else '0')])
        self.global_options.extend(['--float',('1' if float else '0')])
        self.global_options.extend(['--interpolation',interpolation])
        if winsorize:
            self.global_options.extend(['--winsorize-image-intensities',f'[{winsorize[0]},{winsorize[1]}]'])
        self.global_options.extend(['--use-histogram-matching',('1' if histogram_matching else '0')])
    
    def set_initial(self, initialize='identity', fixed=None, moving=None, initial_xfm_moving=True, invert=False):
        initialize = str(initialize).strip()
        if os.path.exists(initialize):
            # Value of initial_xfm_moving=(True)/False toggles
            # --initial-fixed-transform or --initial-moving-transform respectively 
            # --initial-fixed-transform the moving image is pre-moved by the transform 
            # --initial-moving-transform the fixed image is pre-moved by the transform 
#            if not(fixed or moving) or (fixed and moving):
#                raise DiciphrException('For initializing with existing transform, exactly one of initial_fixed, initial_moving need to be True')
            if invert:
                initialize = f'[{initialize},1]'
            if initial_xfm_moving:
                self.initialize_options=['--initial-moving-transform',initialize]
            else:
                self.initialize_options=['--initial-fixed-transform',initialize]
        elif initialize.lower() in ('identity','i'):
            self.initialize_options = []
        else:
            if initialize.lower() in ('center','c','0'):
                # center of image volume (bounding-box)
                init_code = '0'
            elif initialize.lower() in ('mass','m','1'):
                # center of mass 
                init_code = '1'
            elif initialize.lower() in ('origin','o','1'):
                # affine origin points 
                init_code = '2'
            else:
                raise DiciphrException('Unrecognized initial transform option')
            if fixed is None or moving is None:
                raise DiciphrException(f'Fixed and moving images for initialization are required when using initialization method {initialize}')
            self.initialize_options=['--initial-moving-transform',f'[{fixed},{moving},{init_code}]']
    
    def add_stage(self, fixed_images, moving_images, transform, metric, convergence, shrink_factors, smoothing_sigmas,
                    restrict_deformation=None, metric_params=[], metric_weights=[1],
                    samplingStrategy='Regular', samplingPercentage=0.25, transform_params=[],
                    smoothing_unit='vox', convergence_threshold='1e-6', convergence_window=10):
        # fixed - list of fixed filenames of length N 
        # moving - list of fixed moving filenames of length N 
        # metric_weights - list of floats of length N 
        # transform - str 
        # metric - str 
        # convergence - list of ints of length M
        # shrink_factors - list of ints of length M
        # smoothing_sigmas - list of ints of length M
        # metric_params - list of parameters, length depends on metric and will be validated
        # transform_params - list of parameters, length depends on metric and will be validated
        # samplingStrategy - one of "Regular" "Random" "None" 
        # samplingStrategy - float in [0,1]
        # convergence_threshold - float
        # convergence_window - int
        transform, transform_params = self._validate_transform_params(transform, transform_params)
        transform_params = ','.join(map(str,transform_params))
        transform_cmd = ['--transform', f'{transform}[{transform_params}]']
        if restrict_deformation is not None:    
            transform_cmd.extend(['-g', str(restrict_deformation)])
        metric_cmd = []
        if not(len(fixed_images) == len(moving_images)):
            raise DiciphrException("Number of fixed and moving images must be equal")
        if len(metric_weights)<len(fixed_images) and len(metric_weights)==1:
            metric_weights *= len(fixed_images)
        for fixed, moving, weight in zip(fixed_images, moving_images, metric_weights):
            metric, metric_params = self._validate_metric_params(metric, metric_params, samplingStrategy, samplingPercentage)
            metric_params = ','.join(map(str,metric_params))
            metric_cmd.extend(['--metric', f'{metric}[{fixed},{moving},{weight},{metric_params}]'])
        if not(len(convergence) == len(shrink_factors) == len(smoothing_sigmas)):
            raise DiciphrException("Iterables convergence, shrink_factors, and smoothing_sigmas do not match in length")
        # multi-stage options 
        convergence_cmd = ['--convergence', '['+','.join([
                    'x'.join(map(str,map(int,convergence))), 
                    str(convergence_threshold),
                    str(convergence_window)
                    ])+']']
        shrink_factors_cmd = ['--shrink-factors', 'x'.join(map(str,map(int,shrink_factors)))]
        smoothing_cmd = ['--smoothing-sigmas', 'x'.join(map(str,map(int,shrink_factors)))+smoothing_unit]
        self.stages.append({
            'transform':transform,
            'cmd_array':transform_cmd+metric_cmd+convergence_cmd+shrink_factors_cmd+smoothing_cmd 
        })
    
    def _validate_transform_params(self, transform, transform_params):
        if transform in ('Rigid', 'Affine', 'CompositeAffine', 'Similarity', 'Translation'):
            default_params=[0.1]
        elif transform == 'SyN':
            default_params=[0.1,3,0]
        elif transform == 'BSplineSyN':
            default_params=[0.1,1,26,3]
        else: 
            raise DiciphrException("Unrecognized ANTs transform type: {0}".format(transform))
        if len(transform_params) == 0:
            return (transform, default_params)
        elif len(transform_params) == len(default_params):
            return (transform, transform_params)
        else:
            raise DiciphrException("Supported ANTs transform {0} cannot accept {1} out of {2} parameters".format(transform, len(transform_params), len(default_params)))
    
    def _validate_metric_params(self, metric, metric_params, samplingStrategy, samplingPercentage):
        if metric in ('MI', 'Mattes'):
            default_params=[32]
        elif metric == 'CC':
            default_params=[4]
        elif metric in ('MeanSquares', 'Demons', 'GC'):
            default_params=['NA']
        else: 
            raise DiciphrException("Unrecognized ANTs metric type: {0}".format(metric))
        if samplingPercentage < 0 or samplingPercentage > 1:
            raise DiciphrException("Sampling Percentage must be a float between 0 and 1")
        if len(metric_params) == 0:
            return (metric, default_params+[samplingStrategy,samplingPercentage])
        elif len(metric_params) == len(default_params):
            return (metric, metric_params+[samplingStrategy,samplingPercentage])
        else:
            raise DiciphrException("Supported ANTs metric {0} cannot accept {1} out of {2} parameters".format(metric, len(metric_params), len(metric_params)))
        
    def _build_cmd(self):
        cmd = ['antsRegistration'] + self.global_options + self.outputs + self.initialize_options
        for stage in self.stages:
            cmd += stage['cmd_array']
        self.cmd = cmd 
        return cmd 
        
    def run(self):
        if len(self.stages)==0: 
            raise DiciphrException('No stages defined, cannot build antsRegistration call')
        self._build_cmd()
        return ExecCommand(self.cmd).run()

def ants_apply_transforms(input_filename, output_filename, reference_filename, transform_filenames, invert=[], interpolation='Linear', bg_value=0):
    '''
    Runs antsApplyTransforms 

    Parameters
    ----------
    input_filename : str
        input nifti image file
    output_filename : str
        output  nifti image file
    reference_filename : str
        reference nifti image file
    transform_filenames : list
        transform files, as a list 
    invert : Optional[list]
        A list of 1s and 0s, that corresponds to transform_filenames: 1 for invert the transform at that position in the list, 0 for not invert. 
    interpolation : Optional[str]
        The interpolation argument, one of 'Linear', 
                        'NearestNeighbor', 
                        'MultiLabel[<sigma=imageSpacing>,<alpha=4.0>]', 
                        'Gaussian[<sigma=imageSpacing>,<alpha=1.0>]'.
                        'BSpline[<order=3>]',
                        'CosineWindowedSinc',
                        'WelchWindowedSinc',
                        'HammingWindowedSinc',
                        'LanczosWindowedSinc',
                        'GenericLabel[<interpolator=Linear>]'
    bg_value : Optional[float]
        The background fill value of the data, usually 0, but sometimes 1 for e.g. FW VF map. 
    Returns
    -------
    None
    '''
    transform_filenames = force_to_list(transform_filenames)
    if len(invert) == 0:
        invert = [ 0 for tf in transform_filenames ] 
    transform_string = ','.join( [tf if i==0 else '[{},1]'.format(tf) for tf, i in zip(transform_filenames, invert) ])
    cmd = [ 'antsApplyTransforms', 
                '-i', input_filename, 
                '-o', output_filename, 
                '-r', reference_filename, 
                '-t', transform_string,
                '-n', interpolation, 
                '-f', str(bg_value),
                '-v', '1' 
    ]
    ExecCommand(cmd).run() 

def read_ants_affine_transform(transform_file):
    '''
    Call command antsTransformInfo to read an itk transform file.
    '''
    import subprocess
    antsTransformInfo_cmd = which('antsTransformInfo')
    transformInfoOutput = subprocess.check_output([antsTransformInfo_cmd, transform_file], shell=False).decode('ascii')

    matrix = transformInfoOutput.split('Matrix:')[1].split('Offset:')[0]
    matrix = matrix.strip().split('\n')
    matrix = [ a.strip().split() for a in matrix]
    matrix = np.array(matrix).astype(np.float32)

    offset = transformInfoOutput.split('Offset:')[1].split('Center:')[0]
    offset = offset.split('[')[1].split(']')[0].split(',')
    offset = np.array(offset).astype(np.float32)

    center = transformInfoOutput.split('Center:')[1].split('Translation:')[0]
    center = center.split('[')[1].split(']')[0].split(',')
    center = np.array(center).astype(np.float32)

    translation = transformInfoOutput.split('Translation:')[1].split('Inverse:')[0]
    translation = translation.split('[')[1].split(']')[0].split(',')
    translation = np.array(translation).astype(np.float32)

    inverse = np.linalg.inv(matrix)
    
    ret = dict((('Matrix',matrix), 
            ('Offset',offset), 
            ('Center',center), 
            ('Translation',translation), 
            ('Inverse',inverse)))
    return ret
