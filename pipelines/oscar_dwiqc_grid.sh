#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

echo "Executing on: $(hostname)" | tee -a /dev/stderr
echo "Executing in: $(pwd)" | tee -a /dev/stderr
echo "Executing at: $(date)" | tee -a /dev/stderr
echo "Executing   : $0" | tee -a /dev/stderr 
echo "Arguments   : $*" | tee -a /dev/stderr

DICIPHR=##DICIPHR_DIR##
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
module load $DICIPHR/diciphr_module 
module load xvfb 

usage() {
    cat << EOF
Usage: $0 -d DWI -o OUTPUT_PREFIX [ -v VMAX=0.3 ] 
    [ -s SLICETYPE=sagittal ] [ -n SLICENUMBER=60 ] 
    
Required arguments:
    -d  <dwi>      The DWI (or other 4D such as bold) filename
    -o  <path>     The prefix of the output screenshot. ".png" will be appended. 
Optional arguments: 
    -v  <float>    The vmax parameter, range 0+, lower=brighter.     
    -s  <str>      The slice type. Default is sagittal which is good for motion artifacts. 
    -n  <int>      The slice number. Default is 60 which is near the midsagittal line in many 2mm datasets. 
EOF
    exit 1 
}

#### PARAMETERS AND GETOPT  ##################
vmax=0.3
slicetype="sagittal"
slicenumber=60 

while getopts ":d:o:v:s:n:h" opt; do
    case ${opt} in
        h)
            usage ;; 
        d)
            DWI=$OPTARG ;;
        o)
            outbase=$OPTARG ;;
        v)
            vmax=$OPTARG ;;
        s)
            slicetype=$OPTARG ;;
        n) 
            slicenumber=$OPTARG ;; 
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

if [ -z "$DWI" ] || [ -z "$outbase" ]; then 
    log_error "Provide all required options"
    usage 
fi

nt=$(fslhd $DWI | grep dim4 | grep -v pixdim | awk '{print $2}')
figwidth=16 
figheight=$(echo "$nt / 8" | bc)
dpi=300 

log_run oscar.py -i $DWI -o $outbase -s $slicetype -n $slicenumber \
    -v $vmax --dpi $dpi --figsize $figwidth $figheight 
