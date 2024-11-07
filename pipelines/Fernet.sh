#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="Fernet"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 

usage() {
    cat << EOF
Usage: $0  -s subject [ -d DWI ] [ -m mask ] [ -o outdir ]
Required arguments:
    -s    Subject ID, a prefix for output files
Optional arguments:
    -d    DWI image
    -m    DWI mask. 
    -o    Output directory, if not provided will 
              default to $PROJECT_DIR/Protocols/${pipeline_name}
EOF
    exit 1 
}

#### PARAMETERS AND GETOPT  ##################
while getopts ":s:d:m:o:w" opt; do
    case ${opt} in
        s)
            subject=$OPTARG ;;
        d)
            dwi=$OPTARG ;;
        m)
            mask=$OPTARG ;; 
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
# default filenames 
if [ -z "$dwi" ]; then 
    dwi="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_DWI_preprocessed.nii.gz"
fi 
if [ -z "$mask" ]; then 
    mask="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_tensor_mask.nii.gz"
fi 
checkexist $dwi $mask || exit 1 

if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}"
fi
mkdir -p $outdir
setup_logging 
setup_workdir 
log_info "Subject: $subject"
log_info "DWI: $dwi"
log_info "mask: $mask"

log_info "Checking for required input files"
bval=${dwi%.nii.gz}.bval
bvec=${dwi%.nii.gz}.bvec 
checkexist "$dwi" "$mask" "$bval" "$bvec" || exit 1

### BEGIN ###
log_info "Bias correct DWI" 
log_run n4_bias_correction.py -i $dwi -w $mask -o $tmpdir/dwi_N4.nii.gz 

fernet_base=$outdir/${subject}
log_info "Fernet" 
log_run fernet.py -d $tmpdir/dwi_N4.nii.gz \
          -o $fernet_base \
          -m $mask \
          -n 50

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi 
   
log_info "Done" 
