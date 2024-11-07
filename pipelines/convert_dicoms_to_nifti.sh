#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

echo "Executing on: $(hostname)" | tee -a /dev/stderr
echo "Executing in: $(pwd)" | tee -a /dev/stderr
echo "Executing at: $(date)" | tee -a /dev/stderr
echo "Command line: $*" | tee -a /dev/stderr

### DEFINE PROJECT_DIR ##########################
PROJECT_DIR="##PROJECT_DIR##"   
DICIPHR="##DICIPHR_DIR##"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
pipeline_name="convert_dicoms_to_nifti"
module load $DICIPHR/diciphr_module 
module load dcm2niix/1.0.20200331

usage() {
    cat << EOF
Usage: $0 -s subject [ -d dicomdir ] [ -o outputdir ] [ -h, -u ]
EOF
}

dicomdir="dicoms"
outdirflag=""
#### PARAMETERS AND GETOPT  ##################
while getopts ":hus:d:o:" opt; do
    case ${opt} in
        h,u)
            usage
            exit 1 
            ;;
        s)
            subject=$OPTARG ;;
        d)
            dicomdir=$OPTARG ;;
        o)
            outdirflag="-o $OPTARG" ;;
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

cd $PROJECT_DIR 
log_run convert_dicoms_to_nifti.py -d dicoms -s $subject 
