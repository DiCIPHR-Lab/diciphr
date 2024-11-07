#! /bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -e ##PROJECT_DIR##/slurm_output/%j-%x.stderr
#SBATCH -o ##PROJECT_DIR##/slurm_output/%j-%x.stdout

PROJECT_DIR=##PROJECT_DIR##
DICIPHR=##DICIPHR_DIR##
pipeline_name="DeepMedic_to_DTI"
source $PROJECT_DIR/Scripts/pipeline_utils.sh 
# Defaults
outdir="$PROJECT_DIR/Protocols/$pipeline_name"
resolution=0
deepmedicdirs=$PROJECT_DIR/Protocols
regdir=$PROJECT_DIR/Protocols/Registration 

usage() {
    cat << EOF
##############################################
Registers the outputs of the BTpipeline (DeepMedic) to DTI space, for tumor datasets. 
Will resample the output to a desired isotropic resolution (default, 2mm) or without resampling (use -r 0)
This script handles the different ID systems of the BTpipeline (-s, -t) and DiCIPHR pipelines (-i)

Usage: $0 

Required Arguments:
    -s  <str>           Subject ID used for BTpipeline, e.g. ABCD 
    -t  <str>           Timepoint ID used for BTpipeline, e.g. ABCD_2001.01.01
Optional Arguments:
    -i  <str>           ID used for DiCIPHR outputs, if not provided, will default to the timepoint ID used for BTpipeline
    -o  <path>          Path to output directory to be created. Directory "{subjectID}" will be created inside
                            Default: $PROJECT_DIR/Protocols/${pipeline_name}/{subjectID}
    -P  <path>         "Protocols" directory where BTPipeline was run. Default to $PROJECT_DIR/Protocols
    -R  <path>          The registration output directory, containing "Registration_DTI-T1" subdirectory 
                            Default: $PROJECT_DIR/Protocols/Registration 
    -r  <int>           Resample to this isotropic resolution. Default will use native DTI resolution.
##############################################
EOF
}

while getopts ":hus:t:i:P:R:o:r:w" OPT
do
    case $OPT in
        h|u) # help
            usage
            exit 
            ;;
        i) # Subject ID for registration and DTI space outputs 
            subject=$OPTARG
            ;;
        s) # DeepMedic ID, example AAAA 
            sub=$OPTARG
            ;;
        t) # DeepMedic Date, example 2022.02.24
            id=$OPTARG
            ;;
        P) # Contains "6_DeepMedic" etc 
            deepmedicdirs=$OPTARG
            ;;
        R) # Contains "Registratin_DTI-T1" etc 
            regdir=$OPTARG
            ;;
        o)  
            outdir=$OPTARG
            ;;
        r)
            resolution=$OPTARG
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

if [ -z "$subject" ]; then 
    subject=$id
fi 
if [ ! "$(basename $outdir)" == "$subject" ]; then 
    outdir=$outdir/$subject
fi 
log_run mkdir -p $outdir 
setup_logging
setup_workdir 
log_info "subject: $subject"
log_info "BTpipeline subject ID: $sub"
log_info "BTpipeline date ID: $id"
log_info "deepmedicdirs: $deepmedicdirs"
log_info "regdir: $regdir"
log_info "outdir: $outdir"
log_info "resolution: $resolution"

# INPUTS
deepmedic="$deepmedicdirs/6_DeepMedic/$sub/$id/${id}_LPS_rSRI_segmdm.nii.gz"
srimask="$deepmedicdirs/3_Muse/$sub/$id/${id}_LPS_rSRI_maskFinal.nii.gz"
t1ce="$deepmedicdirs/0_ReorientLPS/$sub/$id/${id}_t1ce_LPS.nii.gz"
t1="$deepmedicdirs/0_ReorientLPS/$sub/$id/${id}_t1_LPS.nii.gz"
t2="$deepmedicdirs/0_ReorientLPS/$sub/$id/${id}_t2_LPS.nii.gz"
flair="$deepmedicdirs/0_ReorientLPS/$sub/$id/${id}_flair_LPS.nii.gz"
t1ce_sri_xfm="$deepmedicdirs/2_Registration/$sub/$id/${id}_t1ce_LPS_N4_rSRI.mat"
t1_t1ce_xfm="$deepmedicdirs/2_Registration/$sub/$id/${id}_t1_LPS_N4_rT1ce.mat"
t2_t1ce_xfm="$deepmedicdirs/2_Registration/$sub/$id/${id}_t2_LPS_N4_rT1ce.mat"
flair_t1ce_xfm="$deepmedicdirs/2_Registration/$sub/$id/${id}_flair_LPS_N4_rT1ce.mat"
dtit1regdir="$regdir/Registration_DTI-T1/$subject"
dicoinvwarp="$dtit1regdir/${subject}_dico-1InverseWarp.nii.gz"
dtit1affine="$dtit1regdir/${subject}_DTI-T1-0GenericAffine.mat"
refnii="$dtit1regdir/${subject}_T1-DTI.nii.gz"
probmap_pre="$deepmedicdirs/6_DeepMedic/$sub/$id/predictions/testSession/predictions/${id}_ProbMapClass"
if [ ! -e "$dicoinvwarp" ]; then 
    log_info "Using rigid registration for DTI-T1"
    dtireg_inputs="$dtit1regdir $dtit1affine"
else
    log_info "Using deformable registration for DTI-T1"
    dtireg_inputs="$dtit1regdir $dtit1affine $dicoinvwarp"
fi 
checkexist $t1ce $t1ce_sri_xfm $t1_t1ce_xfm $refnii $dtireg_inputs ${probmap_pre}0.nii.gz || exit 1 

module load $DICIPHR/diciphr_module 
module load greedy/c6dca2e
module load c3d/1.0.0 

log_info "Convert T1ce SRI rigid to ITK"
log_run c3d_affine_tool $t1_t1ce_xfm -oitk $tmpdir/${id}_t1-t1ce-rigid.mat
log_run c3d_affine_tool $t2_t1ce_xfm -oitk $tmpdir/${id}_t2-t1ce-rigid.mat
log_run c3d_affine_tool $flair_t1ce_xfm -oitk $tmpdir/${id}_flair-t1ce-rigid.mat
log_run c3d_affine_tool $t1ce_sri_xfm -oitk $tmpdir/${id}_t1ce-sri-rigid.mat
    
if [ -e "$dicoinvwarp" ]; then 
    # Deformable 
    transform_stack="$dicoinvwarp [$dtit1affine,1] [$tmpdir/${id}_t1ce-sri-rigid.mat,1]"
    transform_stack_nosri="$dicoinvwarp [$dtit1affine,1]"
else
    # Rigid
    transform_stack="[$dtit1affine,1] [$tmpdir/${id}_t1ce-sri-rigid.mat,1]"
    transform_stack_nosri="[$dtit1affine,1]"
fi 

if [ "$resolution" -ne 0 ]; then 
    log_info "Resample DTI space target to $resolution mm" 
    log_run resample_nifti.py -i $refnii -o $tmpdir/ref.nii.gz -r $resolution
    refnii=$tmpdir/ref.nii.gz 
fi 

log_info "Register DeepMedic segmentation to DTI space" 
log_run antsApplyTransforms -d 3 -i $deepmedic -r $refnii -o $outdir/${id}_DeepMedic_DTI.nii.gz \
    -t $transform_stack \
    -v 1 -n GenericLabel 
log_run fslorient -copyqform2sform $outdir/${id}_DeepMedic_DTI.nii.gz

log_run antsApplyTransforms -d 3 -i $srimask -r $refnii -o $tmpdir/${id}_brain_mask.nii.gz \
    -t $transform_stack \
    -v 1 -n GenericLabel 
log_run fslorient -copyqform2sform $tmpdir/${id}_brain_mask.nii.gz

log_info "Register T1CE, T1, T2, FLAIR to DTI" 
log_run antsApplyTransforms -d 3 -i $t1ce -r $refnii -o $tmpdir/${id}_T1CE_DTI.nii.gz \
    -t $transform_stack_nosri \
    -v 1 --float -n Linear 
log_run fslorient -copyqform2sform $tmpdir/${id}_T1CE_DTI.nii.gz
log_run fslmaths $tmpdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_T1CE_DTI.nii.gz $outdir/${id}_T1CE_DTI.nii.gz

log_run antsApplyTransforms -d 3 -i $t1 -r $refnii -o $tmpdir/${id}_T1_DTI.nii.gz \
    -t $transform_stack_nosri $tmpdir/${id}_t1-t1ce-rigid.mat \
    -v 1 --float -n Linear 
log_run fslorient -copyqform2sform $tmpdir/${id}_T1_DTI.nii.gz
log_run fslmaths $tmpdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_T1_DTI.nii.gz $outdir/${id}_T1_DTI.nii.gz

log_run antsApplyTransforms -d 3 -i $t2 -r $refnii -o $tmpdir/${id}_T2_DTI.nii.gz \
    -t $transform_stack_nosri $tmpdir/${id}_t2-t1ce-rigid.mat \
    -v 1 --float -n Linear 
log_run fslorient -copyqform2sform $tmpdir/${id}_T2_DTI.nii.gz
log_run fslmaths $tmpdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_T2_DTI.nii.gz $outdir/${id}_T2_DTI.nii.gz

log_run antsApplyTransforms -d 3 -i $flair -r $refnii -o $tmpdir/${id}_FLAIR_DTI.nii.gz \
    -t $transform_stack_nosri $tmpdir/${id}_flair-t1ce-rigid.mat \
    -v 1 --float -n Linear 
log_run fslorient -copyqform2sform $tmpdir/${id}_FLAIR_DTI.nii.gz
log_run fslmaths $tmpdir/${id}_brain_mask.nii.gz -bin -mul $tmpdir/${id}_FLAIR_DTI.nii.gz $outdir/${id}_FLAIR_DTI.nii.gz
    
log_info "Create edema, tumor masks from DeepMedic segmentation"
log_run fslmaths $outdir/${id}_DeepMedic_DTI.nii.gz -thr 2 -uthr 2 -bin $outdir/${id}_edema.nii.gz 
log_run fslmaths $outdir/${id}_DeepMedic_DTI.nii.gz -bin -sub $outdir/${id}_edema.nii.gz $outdir/${id}_tumor.nii.gz 
    
log_info "Register the probability maps"
for i in 0 1 2 3 4 
do
    log_run antsApplyTransforms -d 3 -i ${probmap_pre}${i}.nii.gz -r $refnii -o $tmpdir/ProbMapClass${i}.nii.gz \
        -t $transform_stack \
        -v 1 --float -n Linear 
    log_run fslorient -copyqform2sform $tmpdir/ProbMapClass${i}.nii.gz
done 
log_run fslmaths $tmpdir/ProbMapClass0.nii.gz -add $tmpdir/ProbMapClass1.nii.gz \
    -add $tmpdir/ProbMapClass2.nii.gz -add $tmpdir/ProbMapClass3.nii.gz \
    -add $tmpdir/ProbMapClass4.nii.gz $tmpdir/ProbSum.nii.gz 
for i in 0 1 2 3 4 
do
    log_run fslmaths $tmpdir/ProbMapClass${i}.nii.gz -div $tmpdir/ProbSum.nii.gz \
        $outdir/${id}_DeepMedic_DTI_ProbMapClass${i}.nii.gz
done

if [ "$cleanup" == "True" ]; then 
    log_info "Clean up" 
    log_run rm -rf $tmpdir 
fi 
