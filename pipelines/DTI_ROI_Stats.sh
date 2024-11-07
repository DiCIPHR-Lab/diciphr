#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
pipeline_name="DTI_ROI_Stats"

usage() {
    cat << EOF
##############################################
This script does the following:
    ROI statistics of scalar maps in Eve (or any other) atlas

Usage: $0 -s <subject> [ options ]

Required Arguments:
    [ -s ]  <str>    Subject IDs as a text file, or a cohort csv file with subject ID in the leftmost column
    [ -c ]  <str>    Name of the scalar. If one of 'FA','MD','AX','RAD','fwVF','fwFA','fwMD','fwAX','fwRAD', 
                     and the filename template not provided, will attempt to locate the scalars in Protocols 
Optional Arguments:
    [ -f ]  <str>    Filename template, a path with {s} in place of the subjectID
    [ -o ]  <str>    Output directory. Default is $PROJECT_DIR/Protocols/DTI_ROI_Stats       
    [ -m ]  <str>    The measure to calculate. Options are mean (Default), std, median, or volume. If volume, scalar is ignored. 
    [ -a ]  <str>    Filename template of the atlas. 
                     Default: $PROJECT_DIR/Protocols/Registration/Registration_DTI-Eve/{s}/{s}_Eve_Labels_to_DTI.nii.gz
    [ -l ]  <str>    CSV lookup for the atlas. Default: $PROJECT_DIR/Templates/EveTemplate/JhuMniSSLabelLookupTable_1.txt
Optional Arguments:
    
##############################################
EOF
    exit 1
}

# DEFAULTS 
lut="$PROJECT_DIR/Templates/EveTemplate/JhuMniSSLabelLookupTable_1.csv"
fernet_default="$PROJECT_DIR/Protocols/Fernet/{s}/{s}_fw" 
tensor_default="$PROJECT_DIR/Protocols/DTI_Preprocess/{s}/{s}_tensor" 
measures=""
atlasfile="$PROJECT_DIR/Protocols/Registration/Registration_DTI-Eve/{s}/{s}_Eve_Labels_to_DTI.nii.gz"
scalarfile=""

while getopts ":hs:f:o:c:a:l:m:w" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) 
            subjectstxt=$OPTARG
            ;;
        f) 
            scalarfile=$OPTARG
            ;;
        o) 
            outdir=$OPTARG 
            ;;
        c)  
            scalar=$OPTARG 
            ;;
        a)
            atlasfile=$OPTARG
            ;;
        l) 
            lut=$OPTARG
            ;;
        m)
            measures="$measures $OPTARG"
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

if [ -z "$subjectstxt" ] || [ -z "$scalar" ]; then 
    log_error "Please provide all required arguments."
    usage
fi 
if [ -z "$measures" ]; then 
    measures="mean"
fi 
# Check scalar is sensible 
if [ -z "$scalarfile" ]; then 
    case $scalar in
        FA) scalarfile="${tensor_default}_FA.nii.gz";;
        MD) scalarfile="${tensor_default}_MD.nii.gz";;
        AX) scalarfile="${tensor_default}_AX.nii.gz";;
        RAD) scalarfile="${tensor_default}_RAD.nii.gz";;
        TR) scalarfile="${tensor_default}_TR.nii.gz";;
        fwFA) scalarfile="${fernet_default}_tensor_FA.nii.gz";;
        fwMD) scalarfile="${fernet_default}_tensor_MD.nii.gz";;
        fwAX) scalarfile="${fernet_default}_tensor_AX.nii.gz";;
        fwRAD) scalarfile="${fernet_default}_tensor_RAD.nii.gz";;
        fwTR) scalarfile="${fernet_default}_tensor_TR.nii.gz";;
        VF) scalarfile="${fernet_default}_volume_fraction.nii.gz";;
        fwVF) scalarfile="${fernet_default}_volume_fraction.nii.gz";;
        *) # Raise error if unrecognized scalar and no filename template 
            log_error "Unrecognized scalar argument and no filename template provided: $scalar" 1>&2 
            usage
            ;;
    esac 
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/DTI_ROI_Stats"
fi
##############################
mkdir -p $outdir 2>/dev/null
setup_logging 
log_info "Check inputs"
checkexist $subjectstxt || exit 1 

log_info "ROI Stats"
log_run roi_stats.py -s $subjectstxt -a $atlasfile -l $lut -f $scalarfile -c $scalar -o $outdir -m $measures 

