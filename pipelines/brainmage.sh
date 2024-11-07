#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=12G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR## 
pipeline_name="brainmage"
module load brainmage/1.0.3
export FSLOUTPUTTYPE=NIFTI_GZ
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
sri24=$PROJECT_DIR/Templates/brainmage/sri24_atlas.nii.gz 

usage() {
    cat << EOF
Usage: $0 -s subject -i image.nii [ -o outdir ] [ -f files.csv ] [ -p params.csv ] 
            [ -m t1 ] [ -1 ] [ -2 ] [ -3 ]
Required arguments:
    -s    Subject ID, a prefix for output files
    -i    Input image (such as t1)
Optional arguments:
    -o    Output directory, if not provided will default to $PROJECT_DIR/Protocols/$pipeline_name
    -f    Files csv, used in batch mode to process a specific batch of files.
    -p    Params csv, used in batch mode to process a specific batch of files.
    -m    Modality used. Default is 't1'. Options are 't1','t1ce','t2','flair'
Batch options:
    -1    Run preprocessing only (registration to SRI24 atlas)
    -2    Run prediction only (brainmage, requires gpu) 
    -3    Run postprocessing only (register masks back to subject space)
EOF
    echo "$*" 1>&2
    exit 1
}
################################
runpreproc="TRUE"
runbrainmage="TRUE"
runpostproc="TRUE"
modality="t1"
while getopts 'huf:p:s:i:m:o:123' OPTION; do
  case "$OPTION" in
    [h,u])
      usage
      ;;
    f)
      filescsv="$OPTARG"
      ;;
    p)
      paramscsv="$OPTARG"
      ;;
    s)
      subject="$OPTARG"
      ;;
    i)
      image="$OPTARG"
      ;;
    o)
      outdir="$OPTARG"
      ;;
    m)
      modality="$OPTARG"
      ;;
    1)
      # preprocessing only
      runbrainmage="FALSE"
      runpostproc="FALSE"
      ;;
    2)
      # predict only
      runpostproc="FALSE"
      runpreproc="FALSE"
      ;;
    3)
      # postprocessing only
      runbrainmage="FALSE"
      runpreproc="FALSE"
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

if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}"
fi
if [ ! -e "$outdir" ]; then 
    log_run mkdir -p $outdir 
fi 
if [ -z "$filescsv" ]; then 
    filescsv=$outdir/files.csv
fi 
if [ -z "$paramscsv" ]; then 
    paramscsv=$outdir/params.csv
fi 
setup_logging
       
if [ "$runpreproc" == "TRUE" ]; then 
    # This mode requires subject and image 
    [ -z "$subject" ] || [ -z "$image" ] && usage "Preprocess mode requires -s and -t" 
    checkexist $image || exit 1
    
    image_reg=$outdir/${subject}_${modality}-to-SRI24-Warped.nii.gz
    if [ ! -e "$image_reg" ]; then 
        ##### Resampling
        image_res=$outdir/${subject}_${modality}_1mm.nii.gz
        if [ ! -f $image_res ]; then
            log_run flirt -in $image -ref $image -out $image_res -nosearch -applyisoxfm 1 -interp trilinear 
        fi

        #### Ants affine registration to sri24 atlas
        antsRegistrationSyNQuick.sh -d 3 -f $sri24 \
            -m $image_res -t r -p f \
            -o $outdir/${subject}_${modality}-to-SRI24- 
    fi 
    # If filescsv wasn't created by batch script, create it now 
    if [ ! -e "$filescsv" ]; then 
        log_info "Building files csv $filescsv"
        echo "Patient_ID_Modality,image_path" > $filescsv
        echo "$subject,$image_reg" >> $filescsv
    fi 
fi 

if [ "$runbrainmage" == "TRUE" ]; then 
    cat > $paramscsv << EOF
# Output directory
results_dir = $outdir
# Directory containing subjects for testing (give path here if using the input_data structure, instead of using the csv input. If using csv input, simply keep the period.)
test_dir = .
# Mode: ma (modality-agnostic), single (only 1 modality is being used), multi (multiple modalities used)
mode = ma
# Whether using a csv file as input (recommended) or not. If False, please provide the test_dir. [True or False]
csv_provided = True
# Path to input csv file when csv_provided=True. If csv_provided=False, please give a period.
test_csv = $filescsv
# The number of modalities for testing
num_modalities = 1
# Type of channels being used:
modalities = ['$modality']
# Set the type of encoder. Options: resunet, fcn, unet
model = resunet
# Set the number of classes. For brain masks, this should be 2 (brain and background).
num_classes = 2
# Set the base filter of the unet
base_filters = 16
EOF
    log_info "Run brainmage"
    log_run brain_mage_run -params $paramscsv -test True -mode MA -dev $CUDA_VISIBLE_DEVICES 
    log_info "Brainmage done" 
fi

if [ "$runpostproc" == "TRUE" ]; then 
    log_info "Warp every mask image back and skull strip in subject space."
    L=$(wc -l $filescsv | awk '{print $1}')
    cat $filescsv | tail -n $((L-1)) | while IFS=, read subject image; do 
        mask=$outdir/${subject}/${subject}_mask.nii.gz
        brain=$outdir/${subject}/${subject}_brain.nii.gz 
        image=$outdir/${subject}_${modality}_1mm.nii.gz
        affine=$outdir/${subject}_${modality}-to-SRI24-0GenericAffine.mat
        mask_subj_space=$outdir/${subject}_${modality}_brain_mask.nii.gz
        skullstripped=$outdir/${subject}_${modality}_brain.nii.gz
        log_info "Postprocessing on: $subject"
        # warp back to native
        log_run antsApplyTransforms -d 3 -i $mask -r $image -o $mask_subj_space -t [$affine,1] -n NearestNeighbor --float -v 1
        # multiply with 1mm image to get skull stripped.
        log_run fslmaths $mask_subj_space -bin -mul $image $skullstripped
        log_run mv -vf $outdir/${subject}_* $outdir/$subject
        # Clean up temp 
        log_run rm -rfv $outdir/Temp/${subject} $outdir/${subject}/${subject}_${modality}-to-SRI24-InverseWarped.nii.gz
        # Rename SRI space results 
        log_run mv -fv $mask $outdir/${subject}/${subject}_SRI24_mask.nii.gz 
        log_run mv -fv $brain $outdir/${subject}/${subject}_SRI24_brain.nii.gz 
    done 
fi
