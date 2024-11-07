#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem-per-cpu=4G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR## 
pipeline_name="brainmage_batch"
module load slurm
module load brainmage/1.0.3
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
sri24=$PROJECT_DIR/Templates/brainmage/sri24_atlas.nii.gz 

usage() {
    cat << EOF
Usage: $0 -i input.txt [ -o outdir ]
Required arguments:
    -i    A text file containing subject ID and path to file in each line, separated by comma ","
Optional arguments:
    -o    Output directory, if not provided will default to $PROJECT_DIR/Protocols/$pipeline_name
    -f    Files csv, provide a unique name if running a specific batch of files
    -p    Params csv, provide a unique name if running a specific batch of files
    -m    Modality used. Default is 't1'. Options are 't1','t1ce','t2','flair'
EOF
    echo "$*" 1>&2 
    exit 1
}
################################
modality="t1"
while getopts 'hui:o:f:p:m:' OPTION; do
  case "$OPTION" in
    [h,u])
      usage
      ;;
    i)
      inputtxt="$OPTARG"
      ;;
    f)
      filescsv="$OPTARG"
      ;;
    p)
      paramscsv="$OPTARG"
      ;;
    m)
      modality="$OPTARG"
      ;;
    o)
      outdir="$OPTARG"
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

if [ -z "$inputtxt" ]; then
    usage
fi 
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
touch $filescsv || usage "Path $filescsv is not writable"
holdj=()
echo "Patient_ID_Modality,image_path" > $filescsv
while IFS=, read subject image 
do
    jid=$(sbatch -J brainmagePre_${subject} --partition=short \
        $PROJECT_DIR/Scripts/brainmage.sh -s $subject \
            -i $image -f $filescsv -m $modality -o $outdir -1
    )
    sleep 1
    holdj+=("$jid")
    image_reg=$outdir/${subject}_${modality}-to-SRI24-Warped.nii.gz
    echo "$subject,$image_reg" >> $filescsv
done < $inputtxt 

if [ -z "$SLURM_JOB_ID" ]; then 
    # batch was run interactively 
    d=$(date +%Y%m%d-%H%M%S)
else
    d=${SLURM_JOB_ID}
fi

# queue the jobs 
hj=$(echo ${holdj[*]} | tr ' ' ':')
jid=$(sbatch --parsable -J brainmageRun_${d} --dependency=afterok:${hj} --gpus=1 --mem-per-cpu=48G \
    $PROJECT_DIR/Scripts/brainmage.sh -f $filescsv -p $paramscsv -m $modality -o $outdir -2 
    )
sleep 1

sbatch -J brainmagePost_${d} --dependency=afterok:${jid} \
    $PROJECT_DIR/Scripts/brainmage.sh -f $filescsv -m $modality -o $outdir -3 
sleep 2

log_info "Submitted jobs for batch. Monitor jobs brainmageRun_${d} and brainmagePost_${d}"