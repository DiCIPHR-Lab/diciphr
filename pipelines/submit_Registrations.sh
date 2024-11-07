#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH --partition=short 
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

echo "Executing on: $(hostname)" | tee -a /dev/stderr
echo "Executing in: $(pwd)" | tee -a /dev/stderr
echo "Executing at: $(date)" | tee -a /dev/stderr
echo "SLURM_JOB_ID: $SLURM_JOB_ID" | tee -a /dev/stderr
echo "Executing   : $0" | tee -a /dev/stderr 
echo "Arguments   : $*" | tee -a /dev/stderr

PROJECT_DIR="##PROJECT_DIR##"   
DICIPHR="##DICIPHR_DIR##"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load slurm

usage() {
    cat << EOF
Submit jobs for Registration pipelines for DTI and T1. 

Usage: $0 -s subject [ -A atlas ] [ -t t1 ] [ -d dwi ] [ -m dwi_mask ] 
        [ -x t1_mask ] [ -o outdir ] [ -r orientation=LPS ]
        [ -p <AP|PA|RL|LR|IS|SI> ] [ -T r ] [ -I 2 ] [ -C ] [ -W ] 
        
Required arguments:
    -s  <str>   Subject ID, a prefix for output files
Optional arguments:
    -A  <str>   Atlas, choose between "Eve" (default), "MNI" or "Yeh"
    -t  <nii>   Skull-stripped T1 image 
    -d  <nii>   DWI image, if not provided will default to 
                    $PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_DWI_preprocessed.nii.gz
    -x  <nii>   Mask in T1 space, if T1 image is not masked.
    -m  <nii>   Mask in DWI space, if DWI image is not masked.
    -o  <dir>   Output directory of the registration pipelines, if not provided will 
                    default to $PROJECT_DIR/Protocols/Registration
    -r  <str>   Orientation string of the data. Default=LPS. Supported orientations: LPS, LAS, RAS
    -p  <str>   Phase encoding direction of the DWI data. Options are none [Default], AP, PA, LR, RL, SI or IS.
                    If -p is provided, transformation type will default to "s"
                    Use this option if running TOPUP was not possible with your DTI data.
    -T  <str>   Transformation type. Options are: a (affine), r (rigid) [Default], s (SyN with restricted deformation)
    -I  <int>   Initialization method of DTI-T1 registraon (1, 2 or 3). Default=2. 
                    0: match by center voxels of each image 
                    1: match by center of mass of each image 
                    2: match by point of origin - assumes DTI and T1 are scanned in the same session and roughly aligned
    -C          Checks that inputs to the pipeline exist, prints commands it would have run, and exits 
    -w          Sets up working directories within output directories and will not delete intermediate files
EOF
    exit 1 
}
#### PIPELINE DEFAULTS ##########
t1_dti_reg_method="r"  # change to "a" for affine, "r" for rigid 
t1_dti_reg_init="2"  # default is origin matching for scans in same session
dti_dico_phaseencoding=""  # change to "" to disable or "-p LR" for LR 
orientation="LPS"
outdir="" 
interactive=""
t1_mask=""
atlas="Eve"
workdirflag=""
################################
while getopts 'hus:d:t:m:o:r:p:x:T:I:A:w' OPTION; do
  case "$OPTION" in
    [h,u])
      usage
      ;;
    s)
      subject="$OPTARG"
      ;;
    A)
      atlas="$OPTARG" 
      ;;
    t)
      t1="$OPTARG"
      ;;
    x)
      t1_mask="$OPTARG"
      ;;
    d)
      dwi="$OPTARG"
      ;;
    m)
      dwi_mask="$OPTARG"
      ;;
    o)
      outdir="$OPTARG"
      ;; 
    r)
      orientation="$OPTARG"
      ;; 
    T)
      t1_dti_reg_method="$OPTARG"
      ;;
    I)
      t1_dti_reg_init="$OPTARG"
      ;;
    p)
      if [ "$OPTARG" == "AP" ] || [ "$OPTARG" == "PA" ] || [ "$OPTARG" == "LR" ] || [ "$OPTARG" == "RL" ] || [ "$OPTARG" == "IS" ] || [ "$OPTARG" == "SI" ]; then 
        dti_dico_phaseencoding="-p $OPTARG"        
      else 
        log_error "Invalid value for DTI phase encoding direction" 
        exit 1
      fi
      ;;
    w) 
      workdirflag="-w"
      ;;
  esac
done

if [ -z "$subject" ]; then
    usage
fi 

if [ ! "$atlas" == "Eve" ]; then 
    if [ "$atlas" == "Yeh" ]; then 
        atlas="MNI"
    elif [ ! "$atlas" == "MNI" ]; then 
        log_error "Atlas argument not recognized: $atlas" && exit 1 
    fi 
fi 
log_info "Atlas: $atlas" 
if [ -n "$dti_dico_phaseencoding" ]; then 
    log_info "User provided phase encoding direction, enabling SyN transform type for DTI-T1 registration"
    t1_dti_reg_method="s"
fi 
if [ "$t1_dti_reg_method" == "s" ] && [ -z "$dti_dico_phaseencoding" ]; then 
    log_error "Option -T s requires that -p be set to one of AP, PA, LR, RL, SI or IS."
    exit 1 
fi 
if [ "$t1_dti_reg_init" != 0 ] && [ "$t1_dti_reg_init" != 1 ] && [ "$t1_dti_reg_init" != 2 ]; then
    log_error "Argument to -I must be 0, 1 or 2"
    exit 1 
fi 
log_info "T1-DTI Registration method: -T $t1_dti_reg_method" 
if [ -n "$dti_dico_phaseencoding" ]; then 
    log_info "DTI phase encoding direction: $dti_dico_phaseencoding" 
fi 

# Necessary filenames
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/Registration"
fi
if [ -z "$t1" ]; then 
    t1="$PROJECT_DIR/Protocols/brainmage/${subject}/${subject}_t1_brain.nii.gz"
fi
t1maskflag=""
if [ -n "$t1_mask" ]; then 
    t1maskflag="-x $t1_mask" 
fi 

if [ -z "$dwi" ]; then 
    dwi="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_DWI_preprocessed.nii.gz"
fi
if [ -z "$dwi_mask" ]; then 
    dwi_mask="$PROJECT_DIR/Protocols/DTI_Preprocess/${subject}/${subject}_tensor_mask.nii.gz"
fi

# atlas filenames 
orn=$(echo "$orientation" | tr '[:upper:]' '[:lower:]')
eve_template_1mm_t1=$PROJECT_DIR/Templates/EveTemplate/JHU_1m_${orn}_T1.nii.gz 
eve_template_2mm_t1=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_T1.nii.gz 
eve_template_wm=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_WMseg.nii.gz
eve_template_mask=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_tensor_mask.nii.gz 
eve_template_labels=$PROJECT_DIR/Templates/EveTemplate/JHU_2m_${orn}_WMPM1.nii.gz 
eve_template_lut=$PROJECT_DIR/Templates/EveTemplate/JhuMniSSLabelLookupTable_1.csv
mni_template_t1=$PROJECT_DIR/Templates/MNI/mni_icbm152_t1_tal_nlin_asym_09a_mask.nii
yeh_atlas_regions=$PROJECT_DIR/Templates/Yeh/regions.txt

# check inputs and set up output filenames 
log_info "Check that required inputs exist" 
dti_t1_affine="$outdir/Registration_DTI-T1/${subject}/${subject}_DTI-T1-0GenericAffine.mat"
checkexist $t1 $dwi $dwi_mask || exit 1 

t1_target_warp="$outdir/Registration_T1-${atlas}/${subject}/${subject}_T1-${atlas}-1Warp.nii.gz"
target_space_fa="$outdir/Registration_DTI-${atlas}/${subject}/${subject}_tensor_FA_to_${atlas}.nii.gz"
if [ "$atlas" == "Eve" ]; then 
    checkexist $eve_template_1mm_t1 $eve_template_2mm_t1 || exit 1 
    checkexist $eve_template_wm $eve_template_mask $eve_template_labels || exit 1 
    checkexist $eve_template_1mm_t1 $eve_template_wm $eve_template_labels $eve_template_lut || exit 1
elif [ "$atlas" == "MNI" ]; then 
    checkexist $mni_template_t1 $yeh_atlas_regions || exit 1
else
    log_error "Atlas argument not recognized: $atlas" && exit 1 
fi 


### SUBMIT DTI-T1 REGISTRATION
if [ ! -e "$dti_t1_affine" ]; then 
    jid1=$(sbatch -J reg_DTI-T1-${subject} --parsable --propagate=NONE \
        $PROJECT_DIR/Scripts/Registration_DTI-T1.sh \
            -s $subject \
            -d $dwi \
            -t $t1 $t1maskflag \
            -o $outdir \
            -m $dwi_mask  \
            -T $t1_dti_reg_method $dti_dico_phaseencoding \
            -I $t1_dti_reg_init $workdirflag
    )
    sleep 1 
    echo "Submitted job $jid1 and added to waiting list"

else
    log_info "Registration_DTI-T1 already run in: $PROJECT_DIR/Protocols/Registration/Registration_DTI-T1/${subject}" 
fi 

### SUBMIT T1-ATLAS REGISTRATION 
if [ ! -e "$t1_target_warp" ]; then 
    jid2=$(sbatch -J reg_T1-${atlas}-${subject} --parsable --propagate=NONE \
        $PROJECT_DIR/Scripts/Registration_T1-${atlas}.sh \
            -s $subject \
            -t $t1 $t1maskflag \
            -o $outdir \
            -r $orientation $workdirflag
    )
    sleep 1 
    echo "Submitted job $jid2 and added to waiting list"

else
    log_info "Registration_T1-${atlas} already run in: $PROJECT_DIR/Protocols/Registration/Registration_T1-${atlas}/${subject}" 
fi 

### SUBMIT DTI-ATLAS REGISTRATION
if [ ! -e "$target_space_fa" ]; then 
    depen=""
    if [ -n "$jid1" ] || [ -n "$jid2" ] ; then
        depen="--dependency=afterok"
        if [ -n "$jid1" ]; then
            depen="${depen}:$jid1"
        fi
        if [ -n "$jid2" ]; then
            depen="${depen}:$jid2"
        fi
    fi

    sbatch -J reg_DTI-${atlas}-${subject} --propagate=NONE $depen \
        $PROJECT_DIR/Scripts/Registration_DTI-${atlas}.sh \
            -s $subject \
            -o $outdir \
            -r $orientation $workdirflag
    sleep 1 
else
    log_info "Registration_DTI-${atlas} already run in: $PROJECT_DIR/Protocols/Registration/Registration_DTI-${atlas}/${subject}" 
fi 
