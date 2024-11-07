#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
pipeline_name="Restbold_Network"
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
    [ -r ]  <path>          Path to Restbold_Preprocess dir
Optional Arguments:
    [ -t ]  <nifti>         T1 image 
    [ -S ]  <path>          Path to Schaefer dir
    [ -a ]  <path>          Flirt registration matrix from bold to structural (T1)
    [ -o ]  <path>          Output directory
    [ -c ]  <int>           Number of regressors used in confound regression. 24 or 36 (Default)
##############################################
EOF
    exit 1
}
atlases=""
creg=36 
while getopts ":hs:t:r:a:o:c:S:w" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) # Subject ID
            subject=$OPTARG
            ;;
        t) 
            t1=$OPTARG 
            ;;
        r)
            restbold_preprocdir=$OPTARG
            ;;
        a)
            affine=$OPTARG
            ;;
        o)
            outdir=$OPTARG
            ;;
        S)
            Schaefer="$OPTARG"
            ;;
        c)  
            creg="$OPTARG"
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
# Restbold Preprocess
if [ -z "$restbold_preprocdir" ]; then 
    log_error "Please provide all required arguments."
    usage
fi

if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}"
fi
# Schaefer atlases 
if [ -z "$Schaefer" ]; then 
    Schaefer="$PROJECT_DIR/Protocols/Schaefer/${subject}"
fi 
# T1 
if [ -z "$t1" ]; then
    t1="$PROJECT_DIR/Protocols/Muse-ss/${subject}/${subject}_brain_muse-ss.nii.gz"
fi 

if [ -z "$affine" ]; then 
    affine=$restbold_preprocdir/coregistration/${subject}_ep2struct.mat
fi 
regressed_bold=$restbold_preprocdir/scrubbing/${subject}_prestats_scrubbed.feat/confound_regress_${creg}EV/${subject}_filtered_func_data_${creg}EV.nii.gz
example_func=$restbold_preprocdir/coregistration/${subject}_example_func_brain.nii.gz

if [ ! "$(basename $outdir)" == "${subject}" ]; then 
    outdir="$outdir/${subject}"
fi 

if [ -z "$atlases" ]; then 
    atlases=$(for parcels in 100 200 300 400 500 600 700 800 900 1000; do echo $Schaefer/${subject}_T1_Schaefer2018_${parcels}_7Networks_ctx_dil.nii.gz; done)
fi 

log_info "Checking inputs" 
checkexist $atlases $t1 $restbold_preprocdir $affine $regressed_bold $example_func || exit 1 

##############################
mkdir -p $outdir 2>/dev/null
setup_logging
setup_workdir
log_info "Subject: $subject"
log_info "Output directory: $outdir"

# BEGIN 
log_run flirt -in $example_func -ref $t1 -applyxfm -init $affine -interp trilinear -out $outdir/ep2struct.nii.gz 

# split up restbold by time 
log_run fslsplit $regressed_bold $tmpdir/funcvol -t 

log_run rm -fv $outdir/*_Timeseries.txt 
for f in $tmpdir/funcvol*
do
    log_run flirt -in $f -ref $t1 -applyxfm -init $affine -interp trilinear -out $tmpdir/reg-$(basename $f)
    for parcels in 100 200 300 400 500 600 700 800 900 1000 
    do
        atlas=$Schaefer/${subject}_T1_Schaefer2018_${parcels}_7Networks_ctx_dil.nii.gz
        timeseries=$outdir/$(basename ${atlas%.nii.gz})_Timeseries.txt
        log_info "Writing to timeseries for $f $atlas" 
        3dROIstats -nomeanout -nzmean -mask $atlas -numROI $parcels \
            -nobriklab -1DRformat $tmpdir/reg-$(basename $f) | tail -n1 | xargs echo | cut -d' ' -f2- >> $timeseries
    done 
    rm -f $tmpdir/reg-$(basename $f) $f 
done

# calculate pearson correlation matrices
pearson_connmat() {
    local timeseries=$1
    local pearson=${timeseries%_Timeseries.txt}_pearson_connmat.txt
    python << EOF
import numpy as np
ts="$timeseries"
p="$pearson"
dat = np.loadtxt(ts)
M = np.corrcoef(dat.transpose())
np.savetxt(p, M, fmt='%0.8f')
EOF
}

for timeseries in $outdir/${subject}_T1_Schaefer2018_*_7Networks_ctx_dil_Timeseries.txt
do
    log_info "pearson_connmat $timeseries" 
    pearson_connmat $timeseries
done

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi 
