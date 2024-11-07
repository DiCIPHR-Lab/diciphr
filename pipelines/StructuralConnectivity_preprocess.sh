#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 2 
#SBATCH --mem-per-cpu=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
pipeline_name="StructuralConnectivity"

usage() {
    cat << EOF
##############################################
This script does the following:
                Runs mrtrix preprocessing for connectivity pipeline 

Usage: $0 -s <subject> -d <dwi> -m <mask> -f <freesurfer> [ options ]

Required Arguments:
    [ -s ]  <string>        Subject ID 
Optional Arguments:
    [ -d ]  <nii>           Path to DWI
    [ -m ]  <nii>           Path to the DWI space mask. 
    [ -f ]  <nii>           Path to the Desikan(Freesurfer) LPS labels file. 
    [ -a ]  <nii>           Path to the affine .mat file from DTI space to T1(Freesurfer) space. 
    [ -i ]  <nii>           Path to the inverse warp in DTI space that matched T1 to EPI distorted DTI. 
    [ -r ]  <int>           Resolution (in mm) of 5tt anatomical image. Default is 1. 
    [ -o ]  <path>          Path to output directory to be created.
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}/Preprocess
    [ -b ]  <string>        Shells, including 0 (for example: -b "0,2000") Default: all shells 
    [ -C ]                  Use CSD fit [Tournier 2007] instead of MSMT-CSD [Jeurissen 2014]
##############################################

EOF
    exit 1
}

#### PARAMETERS ####
nthreads=$SLURM_CPUS_PER_TASK
# Default values 
atlases=()
labeltxts=()
resolution=1
dti_t1_affine=""
dico_invwarp=""
while getopts ":hs:d:m:f:o:b:a:i:Cw" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) # Subject ID
            subject="$OPTARG"
            ;;
        d) # DWI
            dwi="$OPTARG"
            ;;
        m) # mask
            mask="$OPTARG"
            ;;
        f) # freesurfer 
            FreeLabels="$OPTARG"
            ;;
        o) # outdir 
            outdir=${OPTARG%/}
            ;;
        b) # shells
            shells="$OPTARG"
            ;;
        a)
            dti_t1_affine="$OPTARG"
            ;;
        i)
            dico_invwarp="$OPTARG"
            ;;
        C)
            SSST="True"
            ;;
        w)
            workdir="True"
            ;;
        *) # getopts issues an error message
            echo "UNHANDLED OPTION" 1>&2 
            usage
            ;;
    esac
done
if [ -z "$subject" ]; then 
    log_error "Required options not set."
    log_error "subject: $subject"
    usage
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}"
fi
if [ ! "$(basename $outdir)" == "${subject}" ]; then 
    outdir="$outdir/${subject}"
fi 
preprocdir="$outdir/Preprocess"
if [ -z "$dwi" ]; then 
    dwi="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_DWI_preprocessed.nii.gz"
fi 
if [ -z "$mask" ]; then 
    mask="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_tensor_mask.nii.gz"
fi 
if [ -z "$FreeLabels" ]; then 
    FreeLabels="$PROJECT_DIR/Protocols/Freesurfer/${subject}/${subject}_freesurfer_labels.nii.gz"
fi 
if [ -z "$dti_t1_affine" ]; then 
    dti_t1_affine="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_DTI-T1-0GenericAffine.mat"
fi 
if [ -z "$dico_invwarp" ]; then 
    dico_invwarp="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_dico-0InverseWarp.nii.gz"
fi 
##############################
### OUTDIR AND ENVIRONMENT ###
##############################
module load mrtrix/06-06-2017
if [ -n "$JOB_ID" ]; then 
    export MRTRIX_QUIET="yes"
fi 
export FSLOUTPUTTYPE=NIFTI_GZ
mkdir -p $outdir 2>/dev/null
mkdir -p $preprocdir 2>/dev/null
setup_logging 
setup_workdir
log_info "Output directory: $outdir"
log_info "Preprocess directory: $preprocdir"

##############################
######### INPUT DATA #########
##############################
bvec=${dwi%.nii.gz}.bvec
bval=${dwi%.nii.gz}.bval
log_info "Check inputs"
log_info "DWI: $dwi"
log_info "bval: $bval"
log_info "bvec: $bvec"
log_info "mask: $mask"
log_info "dti_t1_affine: $dti_t1_affine"
checkexist $dwi $bval $bvec $mask $FreeLabels $dti_t1_affine || exit 1 

#### BEGIN ####
# Copy the mask to outdir 
cp $mask $preprocdir/mask.nii.gz 
# Resample the mask to resolution

log_info "Resample DTI space to 1mm isotropic." 
target_mask=$tmpdir/mask_1mm.nii.gz
log_run resample_image.py -i $mask -o $target_mask -r 1 -n NearestNeighbor 

desikan=$preprocdir/desikan.nii.gz
# Register the Freesurfer labels, if warp or affine was provided
if [ -e "$dico_invwarp" ] && [ -e "$dti_t1_affine" ]; then 
    log_info "Registering Freesurfer labels to DTI space with transforms: 1) inverse of $dti_t1_affine 2) $dico_invwarp"
    transform_cmd="-t $dico_invwarp [$dti_t1_affine,1]" 
elif [ -e "$dti_t1_affine" ]; then 
    log_info "Registering Freesurfer labels to DTI space with inverse of transform: $dti_t1_affine"
    transform_cmd="-t [$dti_t1_affine,1]" 
else
    log_error "DTI-T1 affine transformation not found."
    exit 1 
fi 
log_run antsApplyTransforms -i $FreeLabels -o $tmpdir/desikan.nii.gz \
    -r $target_mask -n NearestNeighbor $transform_cmd -v 1 --float 
log_run fslmaths $tmpdir/desikan.nii.gz $desikan -odt int 

##############################
#### Create 5TT image ########
##############################
fivett=$preprocdir/5TT_image.nii.gz
fivett_custom=$preprocdir/5TT_image_custom.nii.gz
if [ ! -f "$fivett" ]; then
    log_info "Creating 5tt Image"
    log_run 5ttgen freesurfer $desikan $fivett -nocrop -tempdir $tmpdir
    # create custom 5tt image - move abnormal WM around ventricles from pathological tissue to WM. 
    # split 5tt into five parts in tmp 
    # 0- Cortical grey matter 1- Sub-cortical grey matter 
    # 2- White matter 3- CSF 4- Pathological tissue
    fslsplit $fivett $tmpdir/fivettparts -t 
    fivettparts=($(\ls -1 $tmpdir/fivettparts*))
    # add wm hypointensities into WM 
    log_run fslmaths $desikan -thr 77 -uthr 79 -add ${fivettparts[2]} -bin $tmpdir/fivett_custom_wm.nii.gz 
    # remove those from abnormal 
    log_run fslmaths $desikan -thr 77 -uthr 79 -binv -mul ${fivettparts[4]} -bin $tmpdir/fivett_custom_abn.nii.gz 
    # merge again 
    log_run fslmerge -t $fivett_custom ${fivettparts[0]} ${fivettparts[1]} \
        $tmpdir/fivett_custom_wm.nii.gz ${fivettparts[3]} $tmpdir/fivett_custom_abn.nii.gz 
fi
gmwmi=$preprocdir/gmwmi.nii.gz
if [ ! -f $gmwmi ]; then 
	log_info "Create gmwmi from 5tt image"
	log_run 5tt2gmwmi $fivett $gmwmi
fi

##############################
#### Fit FODs to the Data ####
##############################
sfwm_fod=$preprocdir/sfwm_fod.nii.gz
gm_fod=$preprocdir/gm_fod.nii.gz
csf_fod=$preprocdir/csf_fod.nii.gz

python_flipx_bvec() {
    python << EOF
import numpy as np
bvec_in="$1"
bvec_out="$2"    
np.savetxt(bvec_out, np.loadtxt(bvec_in)*np.array([-1,1,1])[...,None])
EOF
}

if [ ! -f "$sfwm_fod" ]; then 
    #### Extract shells ##########
    if [ -n "$shells" ]; then 
        log_info "Extract shells $shells from DWI" 
        log_run extract_shells_from_multishell_dwi.py -d $dwi -o $tmpdir/dwi.nii.gz -s $shells --logfile $logfile 
        dwi=$tmpdir/dwi.nii.gz
        bval=${dwi%.nii.gz}.bval
        bvec=${dwi%.nii.gz}.bvec
    fi 

    #### Response Functions ######
    voxels=$preprocdir/voxels.nii.gz  #Shows voxel selection for calculating response functions
    sfwm_response=$preprocdir/sfwm_response.txt
    gm_response=$preprocdir/gm_response.txt
    csf_response=$preprocdir/csf_response.txt
    bvec_xflip=$preprocdir/xflip.bvec

    if [ ! -f "$sfwm_response" ]; then
        log_info "Flipping bvec file along x"
        python_flipx_bvec $bvec $bvec_xflip
        log_info "Calculate response functions"
        if [ -z "$SSST" ]; then 
            log_run dwi2response dhollander $dwi \
                                        $sfwm_response $gm_response $csf_response \
                                        -fslgrad $bvec_xflip $bval \
                                        -mask $mask \
                                        -voxels $voxels \
                                        -nthreads $nthreads \
                                        -tempdir $tmpdir
        else
            log_run dwi2response tournier $dwi \
                                        $sfwm_response \
                                        -fslgrad $bvec_xflip $bval \
                                        -mask $mask \
                                        -voxels $voxels \
                                        -nthreads $nthreads \
                                        -tempdir $tmpdir
        fi 
    else
        log_info "White matter response function already exists"
    fi

    log_info "Fit FODs"
    if [ -z "$SSST" ]; then 
        log_run dwi2fod msmt_csd $dwi \
                            $sfwm_response $sfwm_fod \
                            $gm_response $gm_fod \
                            $csf_response $csf_fod \
                            -fslgrad $bvec_xflip $bval \
                            -mask $mask \
                            -nthreads $nthreads 
    else
        log_run dwi2fod csd $dwi \
                            $sfwm_response $sfwm_fod \
                            -fslgrad $bvec_xflip $bval \
                            -mask $mask \
                            -nthreads $nthreads 
    fi 
else
    log_info "White matter FODs already exist"
fi

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up"
    rm -rf $tmpdir 
fi 

log_info "Done"
