#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="Registration_T1-Eve"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 

usage() {
    cat << EOF
Usage: $0  -s subject -t t1 [ -o outdir ] [ -x t1mask ] [ -b ] [ -r orientation=LPS ]
          
Required arguments:
    -s  <str>   Subject ID, a prefix for output files
    -t  <nii>   T1 image
Optional arguments:
    -o  <dir>   Output directory of the registration pipelines, if not provided will 
                    default to $PROJECT_DIR/Protocols/Registration
    -x  <nii>   T1 mask.  If not provided, T1 must be skull-stripped. 
    -r  <str>   Orientation string. Default LPS. Supported orientations: LPS, LAS, RAS  
EOF
    exit 1 
}

#### PARAMETERS AND GETOPT  ##################
bias_correct="False"
orientation="LPS"
while getopts ":s:t:x:o:r:bw" opt; do
    case ${opt} in
        s)
            subject=$OPTARG
            ;;
        t)
            T1=$OPTARG
            ;;
        x)
            t1_mask=$OPTARG
            ;; 
        o)
            outdir=$OPTARG
            ;;
        r)
            orientation=$OPTARG
            ;;
        b)  
            bias_correct="True"
            ;;
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
if [ -z "$T1" ] || [ -z "$subject" ]; then 
    log_error "Provide all required options"
    usage 
fi

orn=$(echo "$orientation" | tr '[:upper:]' '[:lower:]')
eve_template_t1=$PROJECT_DIR/Templates/EveTemplate/JHU_1m_${orn}_T1.nii.gz 
if [ ! -e "$eve_template_t1" ]; then
    log_error "Could not find Eve LPS T1 at $$eve_template_t1"
    exit 1 
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/Registration/${pipeline_name}/${subject}"
else
    outdir="$outdir/${pipeline_name}/${subject}"
fi
mkdir -p $outdir
setup_logging 
setup_workdir
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK
log_info "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
log_info "Subject: $subject"
log_info "T1: $T1"

#################################################
log_info "Checking for required input files"
checkexist "$T1" "$eve_template_t1" || exit 1

#################################################
##########           BEGIN           ############
#################################################
if [ ! -e "${T1toEvePrefix}0GenericAffine.mat" ]; then 
    # Copy masked T1 to tmpdir 
    if [ -z "$t1_mask" ]; then
        t1_mask=$tmpdir/t1_mask.nii.gz 
        log_run fslmaths $T1 -bin $t1_mask 
    else
        log_run fslmaths $t1_mask -bin $tmpdir/t1_mask.nii.gz 
        t1_mask=$tmpdir/t1_mask.nii.gz 
    fi 
    checkexist $t1_mask || exit 1  
    # Bias correction 
    if [ "$bias_correct" == "True" ]; then 
        log_info "Bias correct the T1 and mask"
        N4BiasFieldCorrection -d 3 -i $T1 \
                -o [$tmpdir/t1.nii.gz,$tmpdir/t1_bias.nii.gz] \
                -x $t1_mask \
                -b [150,3] -c [50x50x50x50, 1e-3] \
                -v 1 
    else
        log_info "Skipping bias correction" 
        log_run cp $T1 $tmpdir/t1.nii.gz 
    fi 
    log_run fslmaths $t1_mask -bin -mul $tmpdir/t1.nii.gz $tmpdir/t1_masked.nii.gz
    ##### REGISTRATION #####
    T1toEvePrefix=$outdir/${subject}_T1-Eve-
    log_run antsRegistrationSyN.sh -d 3 -f $eve_template_t1 -m $tmpdir/t1_masked.nii.gz -o $T1toEvePrefix 
else
    log_info "$pipeline_name already run" 
fi 

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi    

log_info "Done" 
