#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="Registration_DTI-T1"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 

usage() {
    cat << EOF
Usage: $0  -s subject -d dwi -m DWI_MASK -t T1 [ -o OUTDIR ] [ -x T1_MASK ] 
          [ -T TRANSFORM_TYPE=r ] [ -p PHASE_ENCONDING=AP ] [ -w workdir ]
Required arguments:
    -s    Subject ID, a prefix for output files
    -d    DWI, bval/bvec files must be named correspondingly, i.e. dwi.nii.gz, dwi.bval, dwi.bvecs
    -t    T1 image
Optional arguments:
    -o    Output directory of the registration pipelines, if not provided will 
              default to $PROJECT_DIR/Protocols/Registration
    -m    dwi space mask
    -x    T1 mask, if T1 passed is not itself skull-stripped, this is required.
    -T    Transformation type. Options are: a (affine), r (Default - rigid),
              s(SyN with restricted deformation), b(B-spline SyN with restricted deformation)
    -p    Phase encoding direction of the dwi data. Options are AP, or LR.
    -I    Initialization method (1, 2 or 3) OR initial transform file (e.g. ITK-Snap transform)
          0: match by center voxels of each image 
          1: match by center of mass of each image 
          2: match by point of origin (Default)
          This script assumes DTI and T1 are scanned in the same session and point of
          origin matching should work well. Try this option if the default does not work. 
    -w    
EOF
    exit 1 
}

#### PARAMETERS AND GETOPT  ##################
phase_encoding="AP"
transform_type="r"
F=0.7
G=0.3
initializemethod=2
initialxfmfile=""
while getopts ":s:d:t:m:o:x:p:T:I:whu" opt; do
    case ${opt} in
        h,u) usage;; 
        s)
            subject=$OPTARG ;;
        d)
            dwi=$OPTARG ;;
        t)
            t1=$OPTARG ;;
        m)
            mask=$OPTARG ;; 
        o)
            outdir=$OPTARG ;;
        x)
            t1_mask=$OPTARG ;;
        p)
            phase_encoding=$OPTARG ;;
        T) 
            transform_type=$OPTARG ;; 
        I)  
            initializemethod=$OPTARG ;;
        w)
            workdir="True" ;; 
        \?)
          log_error "Invalid option: $OPTARG" 1>&2
          usage
          ;;
        :)
          log_error "Invalid option: $OPTARG requires an argument" 1>&2
          usage 
          ;;
    esac
done

##### ENSURE ALL ARGUMENTS ###############
if [ -z "$dwi" ] || [ -z "$t1" ] || [ -z "$subject" ]; then 
    log_error "Provide all required options"
    usage 
fi
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/Registration/${pipeline_name}/${subject}"
else
    outdir="$outdir/${pipeline_name}/${subject}"
fi 
mkdir -p $outdir
setup_logging
setup_workdir
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK
log_info "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
log_info "Subject: $subject"
log_info "dwi: $dwi"
log_info "t1: $t1"
log_info "phase_encoding: $phase_encoding"

if [ "$phase_encoding" == "AP" ] || [ "$phase_encoding" == "PA" ]; then
    restrict_deformation="0.1x1.0x0.1"
elif [ "$phase_encoding" == "LR" ] || [ "$phase_encoding" == "RL" ]; then 
    restrict_deformation="1.0x0.1x0.1"
elif [ "$phase_encoding" == "SI" ] || [ "$phase_encoding" == "IS" ]; then 
    restrict_deformation="0.1x0.1x1.0"
else
    log_error "Phase encoding direction $phase_encoding not recognized"
    exit 1 
fi 
if [ "$transform_type" == "a" ] || [ "$transform_type" == "r" ] || [ "$transform_type" == "s" ]; then   
    log_info "transform_type: $transform_type"
else
    log_error "Transform type $transform_type not recognized"
    exit 1 
fi

#################################################
log_info "Checking for required input files"
checkexist "$t1" "$dwi" || exit 1
if [ -n "$mask" ]; then 
    checkexist "$mask" || exit 1 
fi
if [ ! "$initializemethod" == "0" ] && [ ! "$initializemethod" == "1" ] && [ ! "$initializemethod" == "2" ]; then
    initialxfmfile="$initializemethod"
    log_info "User provided initial transform file" 
    checkexist "$initialxfmfile" || exit 1 
fi 

#################################################
##########           BEGIN           ############
#################################################

# If t1 mask was not provided, assume t1 is skull-stripped already and create it 
if [ -z "$t1_mask" ]; then
    t1_mask=$tmpdir/t1_mask.nii.gz 
    log_run fslmaths $t1 -bin $t1_mask 
fi 
checkexist $t1_mask || exit 1  

# T1 preprocessing steps
log_info "Resample the T1 to 1mm, bias correct the t1 and mask"
log_run resample_image.py -i $t1 -o $tmpdir/t1_1mm.nii.gz -r 1 -n Linear 
log_run resample_image.py -i $t1_mask -o $tmpdir/t1_mask_1mm.nii.gz -m $tmpdir/t1_1mm.nii.gz -n GenericLabel 

if [ ! -e "$tmpdir/t1_bias.nii.gz" ]; then 
    log_run N4BiasFieldCorrection -d 3 -i $tmpdir/t1_1mm.nii.gz  \
        -o [$tmpdir/t1_n4_1mm.nii.gz,$tmpdir/t1_bias.nii.gz] \
        -x $tmpdir/t1_mask_1mm.nii.gz \
        -b [150,3] -c [50x50x50x50, 1e-3] \
        -v 1 
fi 
log_run fslmaths $tmpdir/t1_mask_1mm.nii.gz -bin -mul $tmpdir/t1_1mm.nii.gz $tmpdir/t1_1mm_masked.nii.gz

log_info "Invert t1 contrast to imitation T2 with AFNI" 
if [ ! -e "$tmpdir/t1_masked_imitT2.nii.gz" ]; then 
    perc90=$(3dBrickStat -slow -percentile 90 1 90 -mask $tmpdir/t1_mask_1mm.nii.gz $tmpdir/t1_1mm.nii.gz | awk '{print $2}')
    log_run 3dcalc -overwrite -a $tmpdir/t1_1mm.nii.gz -prefix $tmpdir/t1_thresh.nii.gz -expr maxbelow\($perc90,a\) -float 
    log_run 3dUnifize -echo_edu -prefix $tmpdir/t1_thresh_uni.nii.gz -input $tmpdir/t1_thresh.nii.gz -overwrite 
    perc90=$(3dBrickStat -slow -percentile 90 1 90 -mask $tmpdir/t1_mask_1mm.nii.gz $tmpdir/t1_thresh_uni.nii.gz | awk '{print $2}')
    # log_run 3dcalc -overwrite -a $tmpdir/t1_thresh_uni.nii.gz -b $tmpdir/t1_mask_1mm.nii.gz -prefix $tmpdir/t1_masked_imitT2.nii.gz -expr \(1.1\*${perc90}-a\)\*b+not\(b\)\*a -float 
    log_run 3dcalc -overwrite -a $tmpdir/t1_thresh_uni.nii.gz -b $tmpdir/t1_mask_1mm.nii.gz -prefix $tmpdir/t1_masked_imitT2.nii.gz -expr \(1.1\*${perc90}-a\)\*b -float 
fi 

# DTI Preprocessing steps
log_info "Extract the B0"  
log_run extract_b0.py -d $dwi -o $tmpdir/b0.nii.gz
if [ -z "$mask" ]; then 
    log_info "No DTI space mask provided. Running BET on B0" 
    log_run bet2 $tmpdir/b0.nii.gz $tmpdir/b0 -nm -f 0.2 -g 0.0 
else
    # Binarize in case an FA map or something is passed as mask... 
    log_run fslmaths $mask -bin $tmpdir/b0_mask.nii.gz
fi   
log_info "Calculate FA" 
log_run dti_estimate.py -d $dwi -m $tmpdir/b0_mask.nii.gz -o $tmpdir/orig

log_info "Resample B0 and FA to 1mm" 
log_run resample_image.py -i $tmpdir/b0.nii.gz -o $tmpdir/b0_1mm.nii.gz -r 1 -n Linear 
log_run resample_image.py -i $tmpdir/orig_tensor_FA.nii.gz -o $tmpdir/fa_1mm.nii.gz -m $tmpdir/b0_1mm.nii.gz -n Linear 
log_run resample_image.py -i $tmpdir/b0_mask.nii.gz -o $tmpdir/b0_mask_1mm.nii.gz -m $tmpdir/b0_1mm.nii.gz -n GenericLabel 

log_info "Bias correct the B0" 
if [ ! -e "$tmpdir/bias.nii.gz" ]; then 
    log_run N4BiasFieldCorrection -d 3 -i $tmpdir/b0_1mm.nii.gz \
        -o [$tmpdir/b0_1mm_n4.nii.gz,$tmpdir/bias.nii.gz] \
        -x $tmpdir/b0_mask_1mm.nii.gz \
        -b [150,3] -c [50x50x50x50, 1e-3] \
        -v 1 
fi 
log_info "Mask the B0" 
log_run fslmaths $tmpdir/b0_mask_1mm.nii.gz -bin -mul $tmpdir/b0_1mm_n4.nii.gz $tmpdir/b0_masked.nii.gz

################################
#####   REGISTRATION       #####
dicoPrefix=$outdir/${subject}_dico-
# dico_affine=${dicoPrefix}0GenericAffine.mat
dico_warp=${dicoPrefix}0Warp.nii.gz
dico_invwarp=${dicoPrefix}0InverseWarp.nii.gz
affineDTItoT1Prefix=$outdir/${subject}_DTI-T1-
affineDTItoT1=${affineDTItoT1Prefix}0GenericAffine.mat

b0_target="$tmpdir/b0_masked.nii.gz"
if [ -n "$initialxfmfile" ]; then 
    initialopt="$initialxfmfile"
else
    initialopt="[$tmpdir/t1_masked_imitT2.nii.gz,$b0_target,$initializemethod]"
fi 
if [ "$transform_type" == "s" ]; then
    # Rigid registration DTI to t1  
    log_run antsRegistration \
        --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 \
        --initial-moving-transform "$initialopt" \
        --output [$tmpdir/RIGID-,$tmpdir/RIGID-Warped.nii.gz,$tmpdir/RIGID-InverseWarped.nii.gz] \
        --interpolation Linear --use-histogram-matching 1 --winsorize-image-intensities [0.005,0.995] \
        --transform Rigid[0.1] \
        --metric MI[$tmpdir/t1_masked_imitT2.nii.gz,$tmpdir/b0_masked.nii.gz,1,32,Regular,0.25] \
        --convergence [1000x500x250x100,1e-7,5] \
        --shrink-factors 8x4x2x1 --smoothing-sigmas 4x2x1x0vox
    
    ## SyN registration in DTI space, restricted to AP direction
    log_run antsRegistration \
        --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 \
        --initial-fixed-transform [$tmpdir/RIGID-0GenericAffine.mat,1] \
        --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [0.005,0.995] \
        --convergence [100x70x50x20,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox \
        --transform SyN[0.1,3,0] -g "$restrict_deformation" \
        --output [$dicoPrefix,$tmpdir/fa-Warped.nii.gz] \
        --masks [$tmpdir/t1_mask_1mm.nii.gz,$tmpdir/b0_mask_1mm.nii.gz] \
        --metric MI[$tmpdir/t1_1mm_masked.nii.gz,$tmpdir/fa_1mm.nii.gz,$G,32] \
        --metric MI[$tmpdir/t1_masked_imitT2.nii.gz,$tmpdir/b0_masked.nii.gz,$F,32]
    
    b0_target=$tmpdir/b0_dico.nii.gz
    log_run antsApplyTransforms -d 3 -i $tmpdir/b0_masked.nii.gz -o $b0_target \
        -r $tmpdir/b0_masked.nii.gz -t $dico_warp -n Linear --float -v 1 
fi 

if [ -n "$initialxfmfile" ]; then 
    initialopt="$initialxfmfile"
else
    initialopt="[$tmpdir/t1_masked_imitT2.nii.gz,$b0_target,$initializemethod]"
fi 
log_run antsRegistration \
    --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 \
    --output [$affineDTItoT1Prefix,${affineDTItoT1Prefix}Warped.nii.gz,$tmpdir/InverseWarped.nii.gz] \
    --initial-moving-transform "$initialopt" \
    --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [0.005,0.995] \
    --transform Rigid[0.1] \
    --metric MI[$tmpdir/t1_masked_imitT2.nii.gz,$b0_target,1,32,Regular,0.25] \
    --convergence [1000x500x250x100,1e-6,10] \
    --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
        
## Bring t1 to native res DTI space 
if [ "$transform_type" == "s" ]; then
    log_run antsApplyTransforms -d 3 -i $t1 -o $outdir/${subject}_T1-DTI.nii.gz \
        -r $tmpdir/orig_tensor_FA.nii.gz -t $dico_invwarp [$affineDTItoT1,1] -n Linear --float -v 1 
else
    log_run antsApplyTransforms -d 3 -i $t1 -o $outdir/${subject}_T1-DTI.nii.gz \
        -r $tmpdir/orig_tensor_FA.nii.gz -t [$affineDTItoT1,1] -n Linear --float -v 1 
fi 

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi 

log_info "Done" 
