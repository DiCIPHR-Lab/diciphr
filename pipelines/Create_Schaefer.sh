#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
pipeline_name="Schaefer"
schaefer_dir="$PROJECT_DIR/Templates/Schaefer"
desikan_dir="$PROJECT_DIR/Templates/Desikan"

usage() {
    cat << EOF
##############################################
This script does the following:
    Registers the Schaefer atlas to subject space. 

Usage: $0 -s <subject> [ options ]

Required Arguments:
    [ -s ]  <string>        Subject ID 
Optional Arguments:
    [ -o ]  <path>          Path to output directory to be created. Directory "{subjectID}" will be created inside
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}
    [ -a ]  <file>          Path to the affine .mat file from DTI space to T1(Freesurfer) space. 
    [ -i ]  <file>          Path to the inverse warp in DTI space that matched T1 to EPI distorted DTI. 
    [ -d ]  <file>          A reference map from DTI space, such as the FA map. 
    [ -t ]  <file>          The skull-stripped T1 image.
    [ -f ]  <path>          The freesurfer directory containing directory <subject>. 
##############################################
EOF
    exit 1
}
while getopts ":hs:o:a:d:f:i:t:w" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) # Subject ID
            subject="$OPTARG"
            ;;
        a)
            dti_t1_affine="$OPTARG"
            ;;
        i)
            dico_invwarp="$OPTARG"
            ;;
        d)
            fa="$OPTARG"
            ;;
        t)
            t1="$OPTARG"
            ;;
        f)
            fsdir="$OPTARG"
            ;; 
        o)
            outdir="$OPTARG"
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
    log_error "Please provide all required arguments."
    usage
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}"
fi
if [ ! "$(basename $outdir)" == "${subject}" ]; then 
    outdir="$outdir/${subject}"
fi 
if [ -z "$dti_t1_affine" ]; then 
    dti_t1_affine="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_DTI-T1-0GenericAffine.mat"
fi 
if [ -z "$dico_invwarp" ]; then 
    dico_invwarp="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_dico-0InverseWarp.nii.gz"
fi 
if [ -z "$fsdir" ]; then
    fsdir="$PROJECT_DIR/Protocols/Freesurfer"
fi 
if [ -z "$fa" ]; then
    fa="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_tensor_FA.nii.gz"
fi 
if [ -z "$t1" ]; then
    t1="$PROJECT_DIR/Protocols/Muse-ss/${subject}/${subject}_brain_muse-ss.nii.gz"
fi 

##############################
mkdir -p $outdir 2>/dev/null
setup_logging
setup_workdir 
log_info "Subject: $subject"
log_info "Output directory: $outdir"
mkdir $tmpdir/$subject 
# Link Freesurfer to tmpdir 
# Consider changing this to just edit the Freesurfer directory...
( cd $tmpdir/$subject ; lndir $fsdir/$subject . ) 
export SUBJECTS_DIR=$tmpdir

# BEGIN 
log_info "Check inputs"
checkexist $t1eve_invwarp $t1eve_affine $dti_t1_affine $fa $t1 || exit 1 

if [ -e "$dico_invwarp" ] && [ -e "$dti_t1_affine" ]; then 
    log_info "Registering Freesurfer labels to DTI space with transforms: 1) inverse of $dti_t1_affine 2) $dico_invwarp"
    transform_cmd="-t $dico_invwarp [$dti_t1_affine,1]" 
else 
    log_info "Registering Freesurfer labels to DTI space with inverse of transform: $dti_t1_affine"
    transform_cmd="-t [$dti_t1_affine,1]" 
fi 

log_info "Resample FA to 1mm" 
fa_1mm=$tmpdir/fa_1mm.nii.gz 
log_fun resample_image.py -i $fa -r 1 -o $fa_1mm -n Linear

# Convert fs labels to nifti to compare 
log_run mri_convert --out_orientation LPS \
    $fsdir/$subject/mri/aparc+aseg.mgz $tmpdir/desikan.nii.gz 
log_run resample_image.py -i $tmpdir/desikan.nii.gz -m $t1 -o $tmpdir/desikan-resamp.nii.gz -n NearestNeighbor
    
replace_labels.py -a $tmpdir/desikan-resamp.nii.gz -l $desikan_dir/86_labels.txt \
    -m $desikan_dir/86_labels.txt -o $tmpdir/desikan_gm.nii.gz 

yeo=7 
# Yeo 17 is a permutation of Yeo 7

for parcels in 100 200 300 400 500 600 700 800 900 1000
do 
    Schaefer_T1=$outdir/${subject}_T1_Schaefer2018_${parcels}_${yeo}Networks.nii.gz
    Schaefer_T1_blobs_ctx=$outdir/${subject}_T1_Schaefer2018_${parcels}_${yeo}Networks_ctx_dil.nii.gz
    Schaefer_DTI=$outdir/${subject}_DTI_Schaefer2018_${parcels}_${yeo}Networks.nii.gz
    
    labels_csv="$schaefer_dir/Schaefer_labels_${parcels}_${yeo}.csv"
    log_info "Convert ${yeo}-Network Schaefer $parcels from surface to volume" 
    log_run mris_ca_label -l $SUBJECTS_DIR/${subject}/label/lh.cortex.label \
      ${subject} lh $SUBJECTS_DIR/${subject}/surf/lh.sphere.reg \
      $schaefer_dir/lh.Schaefer2018_${parcels}Parcels_${yeo}Networks.gcs \
      $SUBJECTS_DIR/${subject}/label/lh.Schaefer2018_${parcels}Parcels_${yeo}Networks_order.annot

    # Label subject-specific surface 
    # Commands from ThomasYeoLab github 
    log_run mris_ca_label -l $SUBJECTS_DIR/${subject}/label/rh.cortex.label \
        ${subject} rh $SUBJECTS_DIR/${subject}/surf/rh.sphere.reg \
        $schaefer_dir/rh.Schaefer2018_${parcels}Parcels_${yeo}Networks.gcs \
        $SUBJECTS_DIR/${subject}/label/rh.Schaefer2018_${parcels}Parcels_${yeo}Networks_order.annot
    log_run mri_aparc2aseg --s $subject --o $tmpdir/schaef-${parcels}-${yeo}-2vol.mgz \
        --annot Schaefer2018_${parcels}Parcels_${yeo}Networks_order
        
    # Resample to LPS T1 
    log_run mri_convert --out_orientation LPS \
        $tmpdir/schaef-${parcels}-${yeo}-2vol.mgz $tmpdir/schaef-${parcels}-${yeo}-2vol.nii.gz 
    log_run resample_image.py -i $tmpdir/schaef-${parcels}-${yeo}-2vol.nii.gz -m $t1 \
        -o $tmpdir/schaef-${parcels}-${yeo}-2vol-t1.nii.gz -n NearestNeighbor 
        
    # Select the regions in order of networks 
    log_run replace_labels.py -a $tmpdir/schaef-${parcels}-${yeo}-2vol-t1.nii.gz \
        -o $Schaefer_T1 -c $labels_csv 
        
    # Extract cortical regions - first N hundred ones 
    log_run fslmaths $Schaefer_T1 -uthr $parcels $tmpdir/schaef-${parcels}-${yeo}-t1-ctx.nii.gz 
    
    # Dilate to blobs 
    log_run fslmaths $tmpdir/schaef-${parcels}-${yeo}-t1-ctx.nii.gz -kernel sphere 2 -dilD \
        $tmpdir/schaef-${parcels}-${yeo}-t1-ctx-dil.nii.gz
        
    # Multiply by binarized desikan  
    log_run fslmaths $tmpdir/desikan-resamp.nii.gz -bin -mul $tmpdir/schaef-${parcels}-${yeo}-t1-ctx-dil.nii.gz $Schaefer_T1_blobs_ctx
    
    # Register blobs to 1mm DTI and then extract the labels for connectomes 
    log_run antsApplyTransforms -d 3 \
        -i $Schaefer_T1  \
        -r $fa_1mm \
        -o $Schaefer_DTI \
        $transform_cmd \
        -n NearestNeighbor \
        -v 1  
done 
    
if [ "$cleanup" == "True" ]; then 
    log_info "Clean up" 
    log_run rm -rf $tmpdir 
fi 
