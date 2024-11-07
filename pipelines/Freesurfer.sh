#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
pipeline_name="Freesurfer"

usage() {
    cat << EOF
##############################################
This script does the following:
    Runs Freesurfer, and converts Desikan atlas labels back to Nifti. 

Usage: $0 -s subject -t t1.nii [ -o outdir ]

Required Arguments:
    [ -s ]  <string>        Subject ID 
    [ -t ]  <file>          Path to un-processed T1. 
Optional Arguments:
    [ -o ]  <path>          Path to output directory to be created. Directory "{subjectID}" will be created inside
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}   
##############################################
EOF
    exit 1
}
while getopts ":hus:t:o:w" OPT
do
    case $OPT in
        h|u) # help
            usage
            ;;
        s) # Subject ID
            subject=$OPTARG
            ;;
        t) # T1
            t1=$OPTARG
            ;;
        o) # outdir - delete trailing / if exists 
            outdir=${OPTARG%/}
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
if [ -z "$subject" ] || [ -z "$t1" ] ; then 
    log_error "Please provide all required arguments."
    usage
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}"
fi
if [ ! "$(basename $outdir)" == "${subject}" ]; then 
    outdir="$outdir/${subject}"
fi 

##############################
### OUTDIR AND ENVIRONMENT ###
##############################
module load freesurfer/6.0.0
source /cbica/software/external/freesurfer/centos7/6.0.0/SetUpFreeSurfer.sh
mkdir -p $outdir 2>/dev/null
mkdir -p $outdir/mri/orig 2>/dev/null
setup_logging 
log_info "Subject: $subject"
log_info "Output directory: $outdir"

##############################
######### INPUT DATA #########
##############################
log_info "Check inputs"
log_info "T1: $t1"
checkexist $t1 || exit 1 

#### BEGIN ###################
export SUBJECTS_DIR=$(dirname $outdir)

# Resample T1 
log_run resample_image.py -i $t1 -o $outdir/${subject}_t1_1mm.nii.gz -r 1 -n Linear 

if [ ! -f "$outdir/mri/aparc+aseg.mgz" ]; then 
    log_run mri_convert $outdir/${subject}_t1_1mm.nii.gz $outdir/mri/orig/001.mgz
    ( cd $outdir
    log_run recon-all \
            -log $outdir/recon-all.log \
            -status $outdir/recon-all-status.log \
            -all \
            -s $subject \
            -parallel -openmp $SLURM_CPUS_PER_TASK
    ) 
fi
if [ ! -f "$outdir/${subject}_freesurfer_labels.nii.gz" ]; then 
    log_run mri_convert $outdir/mri/aparc+aseg.mgz $outdir/${subject}_freesurfer_labels.nii.gz \
        --out_orientation LPS -ot nii -rl $outdir/${subject}_t1_1mm.nii.gz -rt nearest
fi 

log_info "Done"