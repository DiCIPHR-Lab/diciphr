#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="Registration_Yeh-DTI"
source $PROJECT_DIR/Scripts/pipeline_utils.sh
module load $DICIPHR/diciphr_module
template_t1=$PROJECT_DIR/Templates/MNI/mni_icbm152_t1_tal_nlin_asym_09a.nii
template_regions=$PROJECT_DIR/Templates/Yeh/regions.txt

usage() {
    cat << EOF
Usage: $0  -s subject [ -o output_dir ] [ -d DTI-T1-regdir ] [ -t T1--regdir] [ -f scalar1 -f scalar2 ... ]
Required arguments:
    -s    Subject ID, a prefix for output files
Optional arguments:
    -o    Output directory of the registration pipelines, if not provided will
              default to $PROJECT_DIR/Protocols/Registration
EOF
    exit 1
}
#### PARAMETERS AND GETOPT  ##################
while getopts ":s:f:o:w" opt; do
    case ${opt} in
        s)
            subject=$OPTARG ;;
        o)
            outdir=$OPTARG ;;
        w)
            workdir="True"
            ;;
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
if [ -z "$subject" ]; then
    log_error "Provide all required options"
    usage
fi
if [ -z "$outdir" ]; then
    outdir="$PROJECT_DIR/Protocols/Registration"
fi
diffusion_regdir="$outdir/Registration_DTI-T1"
t1_regdir="$outdir/Registration_T1-MNI"
dti_dir="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}"
fernet_dir="$PROJECT_DIR/Protocols/Fernet/${subject}"
outdir="$outdir/${pipeline_name}/${subject}"
mkdir -p $outdir
setup_logging 
setup_workdir
log_info "Subject: $subject"
log_info "Diffusion to T1 registration dir: $diffusion_regdir/${subject}"
log_info "T1 to MNI registration dir: $t1_regdir/${subject}"

# input files 
dti_t1_affine=$diffusion_regdir/${subject}/${subject}_DTI-T1-0GenericAffine.mat
dti_warp=$diffusion_regdir/${subject}/${subject}_dico-1Warp.nii.gz
dti_invwarp=$diffusion_regdir/${subject}/${subject}_dico-1InverseWarp.nii.gz
t1_warp=$t1_regdir/${subject}/${subject}_T1-MNI-1Warp.nii.gz
t1_invwarp=$t1_regdir/${subject}/${subject}_T1-MNI-1InverseWarp.nii.gz
t1_affine=$t1_regdir/${subject}/${subject}_T1-MNI-0GenericAffine.mat
# dti and fernet 
fa=$dti_dir/${subject}_tensor_FA.nii.gz
tr=$dti_dir/${subject}_tensor_TR.nii.gz
ax=$dti_dir/${subject}_tensor_AX.nii.gz
rad=$dti_dir/${subject}_tensor_RAD.nii.gz
fwfa=$fernet_dir/${subject}_fw_tensor_FA.nii.gz
fwax=$fernet_dir/${subject}_fw_tensor_AX.nii.gz
fwrad=$fernet_dir/${subject}_fw_tensor_RAD.nii.gz
fwvf=$fernet_dir/${subject}_fw_volume_fraction.nii.gz
# order of transforms for ants
transform_stack_dti_mni="$t1_warp $t1_affine $dti_t1_affine $dti_warp"
transform_stack_mni_dti="$dti_invwarp [$dti_t1_affine,1] [$t1_affine,1] $t1_invwarp"

#################################################
log_info "Checking for required input files"
checkexist $fa $tr $ax $rad $fwfa $fwax $fwrad $fwvf || exit 1 
checkexist $dti_t1_affine $dti_warp $dti_invwarp $t1_warp $t1_invwarp $t1_affine || exit 1
checkexist $template_t1 $template_regions || exit 1

#################################################
##########           BEGIN           ############
#################################################
# Register YEH labels to DTI space and calculate ROI stats.
roistats_prefix=$outdir/${subject}_ROIstats
tmp_prefix=$tmpdir/tmp_roistats
header=$tmpdir/tmp_Header.csv 
echo -n "Subject" > $header 
for sc in FA TR AX RAD fwFA fwAX fwRAD fwVF 
do
    echo -n "$subject" > ${tmp_prefix}_${sc}.csv 
done 
for f in $(cat $template_regions ); do 
    echo "Template mask: $PROJECT_DIR/Templates/Yeh/$f"
    echo -n ",$(basename ${f%.nii.gz})" >> $header
    log_run antsApplyTransforms -d 3 -i $PROJECT_DIR/Templates/Yeh/$f -r $fa \
        -o $outdir/${subject}_Yeh_lin_$(basename $f) \
        -t $transform_stack_mni_dti -n Linear -v 1 --float 
    
    log_run fslmaths $outdir/${subject}_Yeh_lin_$(basename $f) -bin $outdir/${subject}_Yeh_$(basename $f)
    log_run rm -fv $outdir/${subject}_Yeh_lin_$(basename $f)
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $fa 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_FA.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $tr 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_TR.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $ax 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_AX.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $rad 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_RAD.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $fwfa 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_fwFA.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $fwax 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_fwAX.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $fwrad 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_fwRAD.csv 
    echo -n ",$(3dBrickStat -slow -non-zero -median -mask $outdir/${subject}_Yeh_$(basename $f) $fwvf 2>/dev/null | xargs echo | awk '{print $2}')" >> ${tmp_prefix}_fwVF.csv 
    
done 
echo "" >> $header 
for sc in FA TR AX RAD fwFA fwAX fwRAD fwVF 
do
    echo "" >> ${roistats_prefix}_${sc}.csv 
    cat $header ${tmp_prefix}_${sc}.csv > ${roistats_prefix}_${sc}.csv 
done 

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi 

log_info "Done"