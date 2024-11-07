#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="DTI_Preprocess"
source $PROJECT_DIR/Scripts/pipeline_utils.sh
module load $DICIPHR/diciphr_module 

# GETOPTS 
usage() { 
    echo "Run DTI preprocessing with options for topup or no distortion correction" 
    echo "$0 -s subject -d dwi.nii [ -o outdir ] [ -m mask.nii ]"
    echo "      [ -t reversePE_dwi.nii ] [ -p pe_dir=AP ]"
    echo "      [ -T readout_time=0.062 ] [ -w workdir ]" 
}
pe_dir="AP"
readout_time="0.062"
while getopts 'hus:d:o:p:t:T:m:w' OPTION; do
  case "$OPTION" in
    [h,u])
      usage
      exit 1
      ;;
    s)
      subject="$OPTARG"
      ;;
    d)
      dwi="$OPTARG"
      ;;
    t)
      topup="$OPTARG"
      ;;
    m)
      mask="$OPTARG"
      ;;
    o)
      outdir="$OPTARG"
      ;;
    p)
      pe_dir="$OPTARG"
      ;;
    T)
      readout_time="$OPTARG"
      ;;
    w) 
      workdir="True"
      ;;
  esac
done

# Check for required options subject, dwi
if [ -z "$subject" ] || [ -z "$dwi" ]; then
    usage
    echo "Provide all required inputs" 1>&2 
    exit 1 
fi

# Set up necessary directories, log file 
if [ -z "$outdir" ] ; then 
    outdir=$PROJECT_DIR/Protocols/$pipeline_name/${subject}
fi 
mkdir -p $outdir
setup_logging
setup_workdir 
log_info "Subject: $subject"
log_info "DWI: $dwi"
log_info "outdir: $outdir"

# Check all necessary inputs
log_info "Check inputs" 
checkexist "$dwi" || exit 1 
if [ -n "$topup" ]; then
    log_info "topup: $topup"
    log_info "pe_dir: $pe_dir"
    checkexist "$topup" || exit 1 
    topupflag="-t $topup -p $pe_dir"
fi 
if [ -n "$mask" ]; then
    log_info "mask: $mask"
    checkexist "$mask" || exit 1 
    maskflag="-m $mask"
fi 

# Run 
log_run dti_preprocess.py -s $subject \
    -d $dwi -o $outdir -T $readout_time \
    --logfile $logfile --workdir $tmpdir \
    $topupflag $maskflag 

# Remove eddy input and outlier free data
log_run rm -f $outdir/${subject}_eddy.eddy_outlier_free_data* 
log_run rm -f $outdir/${subject}_eddy_input_data* 

if [ "$cleanup" == "True" ]; then 
    log_info "Clean up" 
    log_run rm -rf $tmpdir 
fi 
