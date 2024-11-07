# UTILITY FUNCTIONS USED BY DICIPHR PIPELINE SCRIPTS 

log_info() {
    local timestamp=$(date '+%Y%m%d-%H:%M:%S')
    if [ -f "$logfile" ]; then
        echo "$timestamp [INFO ] $*" >> $logfile
    fi
    if [ -z "$quiet" ]; then
        echo "$timestamp [INFO ] $*" 
    fi
}

log_error() {
    local timestamp=$(date '+%Y%m%d-%H:%M:%S')
    if [ -f "$logfile" ]; then
        echo "$timestamp [ERROR] $*" >> $logfile
    fi
    if [ -z "$quiet" ]; then
        echo "$timestamp [ERROR] $*" 1>&2 
    fi
}

log_run() {
    _prev_run_time=$_current_run_time
    _current_run_time=$(date +%s)
    local cmd=$*
    local elapsed=
    [ -n "$_prev_run_time" ] && elapsed=$(echo "$_current_run_time - $_prev_run_time" | bc)
    log_info "$cmd"
    $cmd
    local rc="$?"
    if [ "$rc" -ne 0 ] ; then
        log_error "Command $cmd exited with status [ $rc ]. Elapsed time $elapsed"
        return $rc
    else
        log_info "Command $cmd completed with status [ 0 ]. Elapsed time $elapsed"
        return 0
    fi
}

setup_logging() {
    local date_stamp=$(date +%Y%m%d-%H%M%S)
    if [ -z "$outdir" ]; then 
        logfile="${pipeline_name}_${date_stamp}.log" 
    else    
        logfile="$outdir/${pipeline_name}_${date_stamp}.log"
    fi 
    touch "$logfile"
    log_info "Pipeline: $pipeline_name"
    log_info "args: $*"
    log_info "Executing on: $(hostname)"
    log_info "Executing in: $(pwd)"
    log_info "Executing at: $(date)"
    if [ -n "$SLURM_JOB_ID" ]; then 
        log_info "SLURM_JOB_ID: $SLURM_JOB_ID"
    fi 
}

checkexist() {
    local f=
    for f in $*; do
        if [ -e "$f" ]; then
            log_info $f
        else
            log_error "$f is Missing. "
            return 1
        fi
    done
    log_info "Inputs OK"
}

setup_workdir() {
    if [ -z "$pipeline_name" ]; then
        local pipeline_name="workdir" 
    fi 
    if [ -n "$1" ]; then 
        # user provided directory
        tmpdir="$(mktemp -d "$2/${pipeline_name}.XXXXXX")"
        cleanup="False"
    elif [ "$workdir" == "True" ] && [ -e "$outdir" ]; then 
        # script has an outdir 
        tmpdir="$(mktemp -d "$outdir/${pipeline_name}.XXXXXX")"
        cleanup="False"
    elif [ -e "$TMPDIR" ]; then 
        # use system TMPDIR 
        tmpdir="$(mktemp -d "$TMPDIR/${pipeline_name}.XXXXXX")"
        cleanup="True"
    else
        # use current working directory 
        tmpdir="$(mktemp -d "./${pipeline_name}.XXXXXX")"
        cleanup="True"
    fi 
    echo "$tmpdir" 
}

strip_nifti_ext() { 
    local fn="$1"
    local ext="${fn##*.}"
    case $ext in 
        'nii') echo "${fn%.*}" ;;
        'gz')
            fn="${fn%.*}"
            ext="${fn##*.}"
            if [ "$ext" == "nii" ]; then
                echo "${fn%.*}"
            else
                log_error "Cannot strip NiFTI extension from file: $1" 
            fi 
        ;;
        'hdr') echo "${fn%.*}" ;;
        'img') echo "${fn%.*}" ;;
        *) log_error "Cannot strip NiFTI extension from file: $1" ;;
    esac 
}