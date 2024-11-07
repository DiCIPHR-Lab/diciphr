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
pipeline_name="StructuralConnectivity"

usage() {
    cat << EOF
##############################################
This script does the following:
    Runs mrtrix tractography and connectome with Desikan atlas, 
    optionally Schaefer atlas, or other user-provided atlases, with multiple options

Usage: $0 -s <subject> [ options ]

Required Arguments:
    [ -s ]  <string>        Subject ID 
    
Optional Arguments:
    [ -o ]  <path>          Path to output directory to be created. 
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}
    [ -p ]  <path>          Path to directory for the FOD and 5TT. If the FOD exists, it will not be overwritten. 
                            Use for 2nd runs of the pipeline or if FOD was created elsewhere. 
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}/Preprocess
    [ -S ]  <path>          Path to the directory of Schaefer atlases
                            Default: $PROJECT_DIR/Protocols/Schaefer/{subjectID}
    [ -a ]  <file>          Path to atlas, optionally a pair "file,textfile" with corresponding node order
                            Can be used more than once for multiple atlases. Basename of atlas file will be 
                            stripped of "{subject}_" prefix and file extension for connectome output filename.                         
    [ -g ]  <string>        Tractography algorithm, either probabilistic IFOD2 (Default) or deterministic SD_STREAM
    [ -A ]  <float>         Angle threshold for tractography. Default: 60
    [ -l ]  <float>         Step size for tractography, in mm. Default: 1
    [ -n ]  <value>         Number of seeds per voxel. Default: 250
    [ -P ]                  Enable PFT (backtracking). 
    [ -K ]                  Keep the whole brain track file, in tck format. 
    [ -T ]                  Keep the whole brain track file, converted to .trk format. 
##############################################

EOF
    exit 1
}

#### PARAMETERS ####
nthreads=$SLURM_CPUS_PER_TASK
minlength=25
maxlength=250
# Default values 
algorithm="IFOD2"
count=250
step=1
angle=60
keep_tck="FALSE"
keep_trk="FALSE"
atlases=()
labeltxts=()
schaeferdir=""
PFT=""

while getopts ":hs:a:o:p:g:A:l:n:S:PKTw" OPT
do
    case $OPT in
        h) # help
            usage
            ;;
        s) # Subject ID
            subject=$OPTARG
            ;;
        a) # atlas
            nf=$(echo "$OPTARG" | awk -F "," ' { print NF } ')
            if [ "$nf" -eq 2 ]; then 
                atlases+=("$(echo $OPTARG | cut -d, -f1)")
                labeltxts+=("$(echo $OPTARG | cut -d, -f2)")
            else
                atlases+=("$OPTARG")
                labeltxts+=("")
            fi
            ;;
        o) # outdir - delete trailing / if exists 
            outdir=${OPTARG%/}
            ;;
        p)
            preprocdir=${OPTARG%/}
            ;;
        S) 
            schaeferdir=${OPTARG%/}
            ;;
        g) # algorithm
            algorithm=$OPTARG
            ;;
        A) # angle
            angle=$OPTARG
            ;;
        l) # step
            step=$OPTARG
            ;;
        n) # count
            count=$OPTARG
            ;;
        P)  
            PFT="-backtrack" 
            ;; 
        K)
            keep_tck="TRUE"
            ;;
        T)
            keep_trk="TRUE"
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
if [ -z "$subject" ]; then 
    log_error "Provide all required options."
    usage
fi 
if [ -z "$outdir" ]; then 
    outdir="$PROJECT_DIR/Protocols/${pipeline_name}/${subject}/Connectomes"
fi
if [ -z "$preprocdir" ]; then 
    preprocdir="$(dirname ${outdir%/})/Preprocess"
fi
desikan=$preprocdir/desikan.nii.gz
if [ -z "$atlases" ]; then 
    labeltxts+=("$PROJECT_DIR/Templates/Desikan/86_labels.txt")
    atlases+=("$desikan")
fi 

if [ -n "$schaeferdir" ]; then 
    for order in 100 200 300 400 500 600 700 800 900 1000
    do 
        labeltxts+=("$PROJECT_DIR/Templates/Schaefer/${order}_labels.txt")
        atlases+=("$schaeferdir/${subject}_DTI_Schaefer2018_${order}_7Networks.nii.gz")
    done
fi 

##############################
### OUTDIR AND ENVIRONMENT ###
##############################
module load mrtrix/06-06-2017
if [ -n "$JOB_ID" ]; then 
    export MRTRIX_QUIET="yes"
fi 
export FSLOUTPUTTYPE=NIFTI_GZ
mkdir -p $outdir 2>/dev/null
mkdir -p $preprocdir 2>/dev/null
setup_logging
setup_workdir
log_info "Output directory: $outdir"
log_info "Preprocess directory: $preprocdir"

#### BEGIN ###################
fivett=$preprocdir/5TT_image.nii.gz
fivett_custom=$preprocdir/5TT_image_custom.nii.gz
gmwmi=$preprocdir/gmwmi.nii.gz
sfwm_fod=$preprocdir/sfwm_fod.nii.gz
mask=$preprocdir/mask.nii.gz 

##############################
####   Run Tractography   ####
##############################
tckfile=$tmpdir/${subject}_wb_tracks.tck
trkfile_kept=$outdir/${subject}_wb_tracks.trk 
tckfile_kept=$outdir/${subject}_wb_tracks.tck 

log_info "Count number of voxels in GMWMI mask"
nvoxels=$(fslstats $gmwmi -V | awk '{print $1}')
nseeds=$(echo "$nvoxels * $count" | bc)
log_info "Number of voxels in GMWMI: $nvoxels" 
log_info "Number of seeds: $nseeds"
echo $nseeds > $outdir/nseeds.txt 

if [ ! -f "$tckfile" ]; then
    log_info "Performing Tractography using $algorithm"
    log_info "Seeding at the WM-GM interface"
	log_run tckgen $sfwm_fod $tckfile  \
        -nthreads $nthreads \
        -algorithm $algorithm \
        -minlength $minlength \
        -maxlength $maxlength \
        -step $step -angle $angle \
        -act $fivett_custom $PFT \
        -seed_gmwmi $gmwmi -seeds $nseeds \
        -mask $mask
fi

if [ "$keep_tck" == "TRUE" ]; then 
    log_info "Copying whole brain .tck file to output directory."
    log_run cp -LRfv $tckfile $tckfile_kept 
elif [ "$keep_trk" == "TRUE" ]; then
    log_info "Creating whole brain .trk file in output directory."
    log_run convert_tracks.py -i $tckfile -o $trkfile_kept -a $mask
fi 

sift_weights=$tmpdir/sift_weights.txt
if [ ! -f "$sift_weights" ]; then
	log_info "Calculating SIFT weights - sift2"
	log_run tcksift2 $tckfile  \
        $sfwm_fod \
        $sift_weights \
        -nthreads $nthreads 
fi  

nAtlases=${#atlases[@]}
for (( i=0; i<${nAtlases}; i++ ))
do
    atlas=${atlases[$i]}
    labeltxt=${labeltxts[$i]}
    log_info "Creating connectome using $atlas $labeltxt..."
    connectome_basename=$(basename $(strip_nifti_ext $atlas) | sed s/"${subject}_"/""/g)
    atlas_reordered=$tmpdir/${connectome_basename}_reordered.nii.gz 
    if [ -n "$labeltxt" ]; then
        log_run replace_labels.py -a $atlas -l $labeltxt -o $atlas_reordered --order 
    else
        # sometimes atlas will not be sequential and no txt file provided.. 
        log_run replace_labels.py -a $atlas -o $atlas_reordered --order 
    fi      
    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat.txt \
        -symmetric -zero_diagonal -force -nthreads $nthreads

    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat_nodevol.txt \
        -scale_invnodevol \
        -symmetric -zero_diagonal -force -nthreads $nthreads

    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat_invlength.txt \
        -scale_invlength \
        -symmetric -zero_diagonal -force -nthreads $nthreads

    ### Calculating the connectomes with sift weights applied
    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat_sift2.txt \
        -tck_weights_in $sift_weights \
        -symmetric -zero_diagonal -force -nthreads $nthreads

    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat_sift2_nodevol.txt \
        -scale_invnodevol \
        -tck_weights_in $sift_weights \
        -symmetric -zero_diagonal -force -nthreads $nthreads

    log_run tck2connectome $tckfile \
        $atlas_reordered ${outdir}/${subject}_${connectome_basename}_connmat_sift2_invlength.txt \
        -scale_invlength \
        -tck_weights_in $sift_weights \
        -symmetric -zero_diagonal -force -nthreads $nthreads
done

# other stuff to do before deleting tck file 
log_info "tckstats on track file "
log_run tckstats $tckfile -nthreads $nthreads > ${outdir}/tckstats.txt

log_info "TDI of endpoints"
log_run tckmap -nthreads $nthreads -template $gmwmi -ends_only $tckfile ${outdir}/tdi_endpoints.nii.gz

log_info "TDI raw"
log_run tckmap -nthreads $nthreads -template $gmwmi $tckfile ${outdir}/tdi.nii.gz

log_info "TDI sift weighted"
log_run tckmap -nthreads $nthreads -template $gmwmi -tck_weights_in $sift_weights $tckfile ${outdir}/tdi_sift2.nii.gz

if [ "$cleanup" == "True" ]; then 
    log_info "Cleaning up"
    rm -rf $tmpdir 
fi 

log_info "Done"
