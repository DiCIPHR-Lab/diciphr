#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=short 
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

echo "Executing on: $(hostname)" | tee -a /dev/stderr
echo "Executing in: $(pwd)" | tee -a /dev/stderr
echo "Executing at: $(date)" | tee -a /dev/stderr
echo "SLURM_JOB_ID: $SLURM_JOB_ID" | tee -a /dev/stderr
echo "Command line: $0 $*" | tee -a /dev/stderr

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load slurm
pipeline_name="StructuralConnectivity_mrtrix"

usage() {
    cat << EOF
Submit jobs for the Structural Connectivity pipeline using mrtrix tools.

Usage: $0 -s <subject> [ options ]

Required Arguments:
    -s  <str>    Subject ID 
Optional Arguments:
    -t  <nii>   Path to Skull-stripped T1. 
    -d  <nii>   Path to DWI
    -m  <nii>   Path to the DWI space mask. 
    -f  <path>  The freesurfer directory containing output for the subject 
    -a  <file>  Path to the affine .mat file from DTI space to T1(Freesurfer) space. 
    -i  <nii>   Path to the inverse warp in DTI space that matched T1 to EPI distorted DTI. 
    -S          If provided, generate Schaefer atlases and pass to connectome script. 
    -C          If provided, checks that inputs to the pipeline exist and exits. 
    -K          If provided, keep the whole brain track file, in .tck format.
    -T          If provided, keep the whole brain track file, converted to .trk format. 
    -w          Sets up working directories in output directories, keeps intermediate files.
EOF
    exit 1
}
schaefer="FALSE"
checkin="FALSE"
keep=""
workdirflag=""
while getopts ":hs:d:m:f:a:i:t:SKTCw" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) # Subject ID
            subject="$OPTARG"
            ;;
        d) # DWI
            dwi="$OPTARG"
            ;;
        m) # mask
            mask="$OPTARG"
            ;;
        f) # directory Freesurfer/ containing <subject>
            fsdir="$OPTARG" 
            ;;
        a) # mask
            dti_t1_affine="$OPTARG" 
            ;;
        w)
            dico_invwarp="$OPTARG"
            ;;
        t) # t1
            t1="$OPTARG"
            ;;
        S)  
            schaefer="TRUE"
            ;;
        K)
            keep="$keep -K"
            ;;
        T)
            keep="$keep -T"
            ;;
        C)
            checkin="TRUE"
            ;;
        w)
            workdirflag="-w"
            ;;
    esac
done
if [ -z "$subject" ]; then 
    log_error "Please provide all required options."
    usage
fi 
if [ -z "$dwi" ]; then 
    dwi="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_DWI_preprocessed.nii.gz"
fi 
if [ -z "$mask" ]; then 
    mask="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_tensor_mask.nii.gz"
fi 
if [ -z "$fsdir" ]; then 
    fsdir="$PROJECT_DIR/Protocols/Freesurfer"
fi 
fslabels=$fsdir/${subject}/${subject}_freesurfer_labels.nii.gz
if [ -z "$t1" ]; then 
    t1="$PROJECT_DIR/Protocols/brainmage/${subject}/${subject}_t1_brain.nii.gz"
fi 
if [ -z "$dti_t1_affine" ]; then 
    dti_t1_affine="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_DTI-T1-0GenericAffine.mat"
fi 
if [ -z "$dico_invwarp" ]; then 
    dico_invwarp="$PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}/${subject}_dico-0InverseWarp.nii.gz"
fi 

# output of preprocess step: FOD 
outdir=$PROJECT_DIR/Protocols/StructuralConnectivity/${subject}
fod=$outdir/Preprocess/sfwm_fod.nii.gz 

mkdir -pv $outdir 
setup_logging
log_info "subject: $subject"
log_info "dwi: $dwi"
log_info "mask: $mask"
log_info "t1: $t1"
log_info "fslabels: $fslabels"
log_info "dti_t1_affine: $dti_t1_affine"
if [ -e "$dico_invwarp" ]; then 
    log_info "dico_invwarp: $dico_invwarp"
else
    log_info "dico_invwarp: none"
fi 

log_info "Check if required inputs exist" 
checkexist $dwi $mask $t1 $fslabels $dti_t1_affine $dico_invwarp || exit 1  
if [ "$checkin" == "TRUE" ]; then
    log_info "Check passed. Exiting"
    exit 0 
fi 

regs=""
if [ -n "$dti_t1_affine" ]; then 
    regs="$regs -a $dti_t1_affine"
fi
if [ -n "$dico_invwarp" ]; then 
    regs="$regs -w $dico_invwarp"
fi 

wait=""
if [ ! -e "$fod" ]; then 
    jid1=$(sbatch -J "SC-pre_${subject}" --parsable \
        $PROJECT_DIR/Scripts/StructuralConnectivity_preprocess.sh \
            -s $subject -d $dwi -m $mask -f $fslabels $regs $workdirflag
    )
    wait="${wait}:${jid1}"
    sleep 1 
else
    log_info "FOD file found $fod. Preprocess already run." 
fi 

if [ "$schaefer" == "TRUE" ]; then 
    schaeferdir=$PROJECT_DIR/Protocols/Schaefer/$subject 
    if [ ! -e "$schaeferdir" ]; then 
        jid2=$(sbatch -J "Schaefer_${subject}" --parsable \
            $PROJECT_DIR/Scripts/Create_Schaefer.sh \
                -s $subject -f $fsdir -d $mask -t $t1 $regs $workdirflag
        )
        wait="${wait}:${jid2}"
        sleep 1      
    fi  
    
    log_info "Submitting tractography and connectome for Desikan and Schaefer atlases" 
    if [ -n "$wait" ]; then 
        sbatch -J "SC-conn_${subject}" --dependency=afterok$wait \
            $PROJECT_DIR/Scripts/StructuralConnectivity_connectomes.sh \
            -s $subject -S $schaeferdir $keep $workdirflag
        ) 
    else 
        sbatch -J "SC-conn_${subject}" \
            $PROJECT_DIR/Scripts/StructuralConnectivity_connectomes.sh \
            -s $subject -S $schaeferdir $keep $workdirflag
        ) 
    fi 
    sleep 1
else 
    log_info "Submitting tractography and connectome for Desikan atlas" 
    if [ -n "$wait" ]; then 
        sbatch -J "SC-conn_${subject}" --dependency=afterok$wait \
            $PROJECT_DIR/Scripts/StructuralConnectivity_connectomes.sh \
                -s $subject $keep $workdirflag
    else
        sbatch -J "SC-conn_${subject}" \
            $PROJECT_DIR/Scripts/StructuralConnectivity_connectomes.sh \
                -s $subject $keep $workdirflag
    fi 
    sleep 1 
fi 
