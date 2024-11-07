#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="DeepMedic_to_T1"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
module load greedy/c6dca2e

usage() {
    cat << EOF
Usage: $0 -s subject -t sub_timepoint [ -r resample=1 ] 

Post-processing script for BTPipeline; Resample T1 inputs to 1mm isotropic 
(or user-defined resolution) and brings masks from SRI template to subject space. 

Required arguments:
    -s  SubjectID, e.g. ABCD 
    -t  TimepointID, e.g. ABCD_2001.01.01
Optional arguments: 
    -r  Resample the T1 to this isotropic resolution in mm (default 1), or provide 0 to disable 
    -d  "Protocols" directory where BTPipeline was run. Default to $PROJECT_DIR/Protocols 

Example: $0 -s ABCD -t ABCD_2001.01.01 -r 1 
EOF
    exit 1 
}

id=""
sub=""
datadir="$PROJECT_DIR/Protocols"

while getopts ":s:t:d:w" opt; do 
    case ${opt} in 
        h,u) usage;;
        s) 
            sub=$OPTARG;;
        t) 
            id=$OPTARG;;
        d) 
            datadir=$OPTARG;;     
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
if [ -z "$sub" ] || [ -z "$id" ]; then
    log_error "Provide all required options"
    usage
fi

flair_t1ce_xfm="$datadir/2_Registration/$sub/$id/${id}_flair_LPS_N4_rT1ce.mat"
flair_lps="$datadir/0_ReorientLPS/$sub/$id/${id}_flair_LPS.nii.gz"
deepmedic="$datadir/6_DeepMedic/$sub/$id/${id}_LPS_rSRI_segmdm.nii.gz"
t1ce_sri_xfm="$datadir/2_Registration/$sub/$id/${id}_t1ce_LPS_N4_rSRI.mat"
t1ce_lps="$datadir/0_ReorientLPS/$sub/$id/${id}_t1ce_LPS.nii.gz"
t1ce_sri_brain="$datadir/5_SSFinal/$sub/$id/${id}_t1ce_LPS_rSRI_SSFinal.nii.gz"
t1_t1ce_xfm="$datadir/2_Registration/$sub/$id/${id}_t1_LPS_N4_rT1ce.mat"
t1_lps="$datadir/0_ReorientLPS/$sub/${id}/${id}_t1_LPS.nii.gz"
t2_t1ce_xfm="$datadir/2_Registration/$sub/${id}/${id}_t2_LPS_N4_rT1ce.mat"
t2_lps="$datadir/0_ReorientLPS/$sub/${id}/${id}_t2_LPS.nii.gz" 

outdir=$PROJECT_DIR/Protocols/$pipeline_name/$id
mkdir -p $outdir 
setup_logging 
setup_workdir

cp $t1ce_lps $outdir/${id}_T1ce.nii.gz 
t1ce_lps=$outdir/${id}_T1ce.nii.gz 

# bring T1 mask from SRI space to original T1CE space. 
log_run /cbica/software/external/greedy/centos7/c6dca2e/bin/greedy \
    -d 3 \
    -r $t1ce_sri_xfm,-1 \
    -rf $t1ce_lps \
    -ri NN \
    -rm $t1ce_sri_brain $tmpdir/greedyoutput.nii.gz 

log_run fslmaths $tmpdir/greedyoutput.nii.gz -bin $outdir/${id}_brain_mask.nii.gz 

# bring deepmedic outputs from SRI space to original T1CE space. 
log_run /cbica/software/external/greedy/centos7/c6dca2e/bin/greedy \
    -d 3 \
    -r $t1ce_sri_xfm,-1 \
    -rf $t1ce_lps \
    -ri NN \
    -rm $deepmedic $outdir/${id}_deepmedic_segm.nii.gz 

# register T1 to T1ce 
log_run /cbica/software/external/greedy/centos7/c6dca2e/bin/greedy \
    -d 3 \
    -r $t1_t1ce_xfm \
    -rf $t1ce_lps \
    -rm $t1_lps $outdir/${id}_T1_rT1ce.nii.gz

log_run fslmaths $outdir/${id}_brain_mask.nii.gz -bin -mul $outdir/${id}_T1_rT1ce.nii.gz $outdir/${id}_T1_rT1ce_SS.nii.gz 

# register T2 to T1ce 
log_run /cbica/software/external/greedy/centos7/c6dca2e/bin/greedy \
    -d 3 \
    -r $t2_t1ce_xfm \
    -rf $t1ce_lps \
    -rm $t2_lps $tmpdir/${id}_T2_rT1ce.nii.gz 

log_run fslmaths $outdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_T2_rT1ce.nii.gz $outdir/${id}_T2_rT1ce_SS.nii.gz

# register FLAIR to T1ce
log_run /cbica/software/external/greedy/centos7/c6dca2e/bin/greedy \
    -d 3 \
    -r $flair_t1ce_xfm \
    -rf $t1ce_lps \
    -rm $flair_lps $tmpdir/${id}_FLAIR_rT1ce.nii.gz

log_run fslmaths $outdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_FLAIR_rT1ce.nii.gz $outdir/${id}_FLAIR_rT1ce_SS.nii.gz
log_run fslmaths $outdir/${id}_brain_mask.nii.gz -bin -mul $t1ce_lps $outdir/${id}_T1ce_SS.nii.gz 

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up" 
    rm -rf $tmpdir 
fi    