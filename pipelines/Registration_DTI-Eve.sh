#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --partition=short
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR## 
pipeline_name="Registration_DTI-Eve"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 

usage() {
    cat << EOF
Usage: $0  -s subject [ -o outdir ] [ -d dtidir ] [ -f fernetdir ] [ -r orientation=LPS ]
Required arguments:
    -s  <str>   Subject ID, a prefix for output files
Optional arguments:
    -o  <str>   Output directory of the registration pipelines, if not provided will 
                    default to $PROJECT_DIR/Protocols/Registration
    -d  <str>   Directory of DTI preprecessed data. Default: $PROJECT_DIR/Protocols/DTI_Preprocess
    -f  <str>   Directory of DTI FERNET data. Default: $PROJECT_DIR/Protocols/Fernet
    -r  <str>   Orientation string. Default LPS. Supported orientations: LPS, LAS, RAS
EOF
    exit 1 
}
scalars_list=()
dti_dir=""
fernet_dir=""
orientation="LPS"
#### PARAMETERS AND GETOPT  ##################
nthreads=$ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
while getopts ":s:d:f:o:r:w" opt; do
    case ${opt} in
        s)
            subject=$OPTARG ;;
        d) 
            dti_dir=$OPTARG ;;
        f) 
            fernet_dir=$OPTARG ;;
        o)
            outdir=$OPTARG ;;
        r) 
            orientation=$OPTARG ;; 
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

orn=$(echo "$orientation" | tr '[:upper:]' '[:lower:]')
eve_template_t1=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_T1.nii.gz 
eve_template_wm=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_WMseg.nii.gz
eve_template_mask=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_tensor_mask.nii.gz 
eve_template_labels=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_WMPM1.nii.gz 
eve_template_lut=$PROJECT_DIR/Templates/EveTemplate/JhuMniSSLabelLookupTable_1.csv

if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/Registration"
fi
if [ -z "$dti_dir" ]; then 
    dti_dir="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}"
fi 
if [ -z "$fernet_dir" ]; then 
    fernet_dir="$PROJECT_DIR/Protocols/Fernet/${subject}"
fi 
diffusion_regdir="$outdir/Registration_DTI-T1"
t1_regdir="$outdir/Registration_T1-Eve"
outdir="$outdir/${pipeline_name}/${subject}"
mkdir -p $outdir
setup_logging 
setup_workdir
log_info "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
log_info "Subject: $subject"
log_info "Diffusion to T1 registration dir: $diffusion_regdir/${subject}"
log_info "T1 to Eve registration dir: $t1_regdir/${subject}"
if [ -e "$fernet_dir" ]; then 
    log_info "Fernet dir: $fernet_dir"
fi
if [ -e "$dti_dir" ]; then 
    log_info "DTI dir: $dti_dir"
fi 

# Define filenames
dti_t1_affine=$diffusion_regdir/${subject}/${subject}_DTI-T1-0GenericAffine.mat
dti_dicowarp=$diffusion_regdir/${subject}/${subject}_dico-0Warp.nii.gz
dti_dicoinvwarp=$diffusion_regdir/${subject}/${subject}_dico-0InverseWarp.nii.gz
t1_eve_warp=$t1_regdir/${subject}/${subject}_T1-Eve-1Warp.nii.gz
t1_eve_invwarp=$t1_regdir/${subject}/${subject}_T1-Eve-1InverseWarp.nii.gz
t1_eve_affine=$t1_regdir/${subject}/${subject}_T1-Eve-0GenericAffine.mat 
dtispace=$diffusion_regdir/${subject}/${subject}_T1-DTI.nii.gz
fa=$dti_dir/${subject}_tensor_FA.nii.gz
tr=$dti_dir/${subject}_tensor_TR.nii.gz
ax=$dti_dir/${subject}_tensor_AX.nii.gz
rad=$dti_dir/${subject}_tensor_RAD.nii.gz
vf=$fernet_dir/${subject}_fw_volume_fraction.nii.gz
fwfa=$fernet_dir/${subject}_fw_tensor_FA.nii.gz
fwax=$fernet_dir/${subject}_fw_tensor_AX.nii.gz
fwrad=$fernet_dir/${subject}_fw_tensor_RAD.nii.gz
# Order of transforms for ants
if [ -e "$dti_dicoinvwarp" ]; then
    transform_stack_dti_eve="$t1_eve_warp $t1_eve_affine $dti_t1_affine $dti_dicowarp" 
    transform_stack_eve_dti="$dti_dicoinvwarp [$dti_t1_affine,1] [$t1_eve_affine,1] $t1_eve_invwarp" 
else
    transform_stack_dti_eve="$t1_eve_warp $t1_eve_affine $dti_t1_affine" 
    transform_stack_eve_dti="[$dti_t1_affine,1] [$t1_eve_affine,1] $t1_eve_invwarp" 
fi 

#################################################
log_info "Checking for required input files"
if [ -n "${scalars_list[0]}" ]; then 
    # if user provided scalars as "file1 file2 file3" separate them by spaces 
    scalars_list=(${scalars_list[@]})
    checkexist ${scalars_list[@]}|| exit 1
fi 
checkexist $dti_t1_affine $t1_eve_warp $t1_eve_affine || exit 1 
checkexist $eve_template_t1 || exit 1 
checkexist $eve_template_wm $eve_template_mask || exit 1 
checkexist $eve_template_labels || exit 1 

#################################################
##########           BEGIN           ############
#################################################
reg_scalar() {
    local scalar="$1"
    local outblur="$outdir/$(basename ${scalar%.nii.gz}_to_Eve_brain_4fwhm.nii.gz)"
    if [ ! -e "$outblur" ]; then 
        log_run antsApplyTransforms -d 3 -i $scalar \
            -o $outdir/$(basename ${scalar%.nii.gz}_to_Eve.nii.gz) \
            -r $eve_template_t1 -t $transform_stack_dti_eve \
            -n Linear --float -v 1
        # blur in WM 
        log_run 3dmerge -1filter_blur 4 -1fmask $eve_template_wm \
            -prefix $outdir/$(basename ${scalar%.nii.gz}_to_Eve_WM_4fwhm.nii.gz) \
            $outdir/$(basename ${scalar%.nii.gz}_to_Eve.nii.gz)
        # blur in whole mask  
        log_run 3dmerge -1filter_blur 4 -1fmask $eve_template_mask \
            -prefix $outdir/$(basename ${scalar%.nii.gz}_to_Eve_brain_4fwhm.nii.gz) \
            $outdir/$(basename ${scalar%.nii.gz}_to_Eve.nii.gz)
    fi 
}
# Register scalars to Eve space 
[ -e "$fa" ] && reg_scalar $fa 
[ -e "$tr" ] && reg_scalar $tr 
[ -e "$ax" ] && reg_scalar $ax 
[ -e "$rad" ] && reg_scalar $rad 
[ -e "$vf" ] && reg_scalar $vf 
[ -e "$fwfa" ] && reg_scalar $fwfa 
[ -e "$fwax" ] && reg_scalar $fwax 
[ -e "$fwrad" ] && reg_scalar $fwrad 

# Register Eve labels to DTI space. 
atlasfn="$outdir/${subject}_Eve_Labels_to_DTI.nii.gz"
if [ ! -e "$outdir/${subject}_Eve_Labels_to_DTI.nii.gz" ]; then 
    log_run antsApplyTransforms -d 3 -i $eve_template_labels -r $dtispace \
        -o $atlasfn -t $transform_stack_eve_dti -n NearestNeighbor -v 1 --float 
fi       
# Create some useful links 
[ ! -e "$outdir/$(basename $eve_template_t1)" ] && log_run ln -s $eve_template_t1 $outdir 
[ ! -e "$outdir/$(basename $eve_template_labels)" ] && log_run ln -s $eve_template_labels $outdir 
[ ! -e "$outdir/$(basename ${scalars_list[0]})" ] && log_run cp -v ${scalars_list[0]} $outdir 

log_info "ROI stats for all scalars" 
header=$tmpdir/header.csv 
echo -n "Subject" > $header
cat $eve_template_lut | tail -n 176 | while IFS=, read label roi name tissue hemi vol note 
do
    label=$(printf "%03d" $label)
    echo -n ",${label}_${roi}" >> $header 
done 
echo "" >> $header 
roistats() { 
    local fn=$1
    local measure=$2
    local output=$3
    cat $header > $output 
    log_info "3dROIstats -mask $atlasfn -nomeanout -nobriklab -1DRformat -nz${measure} $fn"
    echo -n "$subject," >> $output
    3dROIstats -mask $atlasfn -nomeanout -nobriklab -1DRformat -nz${measure} $fn | tail -n1 | xargs echo | cut -d' ' -f2- | sed s/' '/','/g >> $output 
}
for measure in median mean sigma; do 
    [ -e "$fa" ] && roistats $fa $measure $outdir/${subject}_ROIstats_${measure}_FA.csv
    [ -e "$tr" ] && roistats $tr $measure $outdir/${subject}_ROIstats_${measure}_TR.csv
    [ -e "$ax" ] && roistats $ax $measure $outdir/${subject}_ROIstats_${measure}_AX.csv
    [ -e "$rad" ] && roistats $rad $measure $outdir/${subject}_ROIstats_${measure}_RAD.csv
    [ -e "$vf" ] && roistats $vf $measure $outdir/${subject}_ROIstats_${measure}_fwVF.csv
    [ -e "$fwfa" ] && roistats $fwfa $measure $outdir/${subject}_ROIstats_${measure}_fwFA.csv
    [ -e "$fwax" ] && roistats $fwax $measure $outdir/${subject}_ROIstats_${measure}_fwAX.csv
    [ -e "$fwrad" ] && roistats $fwrad $measure $outdir/${subject}_ROIstats_${measure}_fwRAD.csv
done 

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi 
   
log_info "Done" 
