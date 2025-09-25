# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, sys
import subprocess, threading, signal
import logging, tempfile, shutil
import json
import atexit 
import argparse 
from datetime import datetime
from socket import gethostname
from getpass import getuser
import nibabel as nib
import numpy as np

# Ensure logging is safely shut down at program exit
atexit.register(logging.shutdown)

##############################################
############     EXCEPTIONS     ##############
##############################################
class DiciphrException(Exception):
    pass
    
##############################################
############   FILE FUNCTIONS   ##############
##############################################
def is_writable(filename):
    '''Returns true if file has write access.'''
    return os.access(filename,os.W_OK)
    
def make_dir(path, recursive=False, pass_if_exists=False):
    """
    Make a directory at the given path.
    
    Parameters:
    - path (str): The directory path to create.
    - recursive (bool): If True, create intermediate directories as needed.
    - pass_if_exists (bool): If True, do not raise an error if the directory already exists.
    - verbose (bool): If True, print status messages.
    
    Returns:
    - str: The path of the created (or existing) directory.
    """
    try:
        if recursive:
            os.makedirs(path, exist_ok=pass_if_exists)
        else:
            if pass_if_exists and os.path.exists(path):
                logging.debug(f"Directory already exists: {path}")
            else:
                os.mkdir(path)
        logging.info(f"Directory created: {path}")
    except FileExistsError:
        if not pass_if_exists:
            raise
        logging.debug(f"Directory already exists and was skipped: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create directory '{path}': {e}")
    
    return path

def make_temp_dir(directory=None, prefix='tmp'):
    '''Creates and returns a path to a unique working directory.

    If `directory` is None:
        - Use TMPDIR if set, else raise EnvironmentError

    Parameters
    ----------
    directory : Optional[str]
        Directory inside which to create a new tempdir
    prefix : Optional[str]
        Prefix of the new tempdir, default 'tmp'

    Returns
    -------
    str
        The newly created tempdir.
    '''
    if directory is None:
        directory = get_diciphr_tmpdir()
        if directory is None:
            raise EnvironmentError("No temporary directory specified. Environmental variable TMPDIR must be set.")    

    # Ensure the base directory exists
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.exception(f"Failed to create base temp directory {directory}: {e}")
        raise

    # Try to create a unique temp directory inside it
    jobid = os.environ.get('SLURM_JOB_ID','')
    if jobid:
        prefix = prefix+'_'+jobid 
    try:
        from uuid import uuid4
        workdir = tempfile.mkdtemp(prefix=prefix, dir=directory, suffix=str(uuid4()))
    except Exception as e:
        logging.warning(f"Failed to create temp dir with UUID suffix: {e}")
        try:
            workdir = tempfile.mkdtemp(prefix=prefix, dir=directory)
        except Exception as e2:
            logging.exception(f"Failed to create temp dir without UUID suffix: {e2}")
            raise

    logging.info(f"Created temporary directory {workdir}")
    return workdir

class TempDirManager:
    def __init__(self, directory=None, prefix='tmp', delete=True):
        self.tmpdir = make_temp_dir(directory, prefix)
        if get_diciphr_tmpdir():
            # User provided --workdir option - do not clean up 
            self.delete = False
        else:
            self.delete = delete 
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        
    def path(self):
        return self.tmpdir 
    
    def cleanup(self):
        if self.tmpdir and os.path.exists(self.tmpdir) and self.delete:
            try:
                logging.info(f"Deleting temp dir: {self.tmpdir}")
                shutil.rmtree(self.tmpdir)
            except Exception as e:
                logging.warning(f"Failed to delete temp dir {self.tmpdir}: {e}")
            finally:
                self.tmpdir = None 
        elif not self.delete:
            logging.info(f"Preserving temp dir: {self.tmpdir}")

def random_string(length=16):
    import random
    import string 
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))
    
def remove_file(*args):
    for filename in args:
        os.remove(filename)
                
def remove_dir(recursive=False,*args):
    '''Remove a directory, optionally recursively.'''
    for directory in args:
        if recursive:
            shutil.rmtree(directory)
        else:
            os.rmdir(directory)
            
def find_all_files_in_dir(directory, followlinks=True):
    '''Recursively find all files in a directory.'''
    files_list=sum([[os.path.join(root, name) for name in files] for root, dirs, files in os.walk(directory, followlinks=followlinks)],[])
    logging.debug('Files: {}'.format(files_list))
    return files_list

def read_json_file(filepath):    
    """
    Loads a JSON file and returns its contents as a dictionary.

    Parameters:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Dictionary representation of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    with open(filepath, 'r', encoding='utf-8') as fid:
        return json.load(fid)
    
##############################################
####  SCRIPT LOGGING/ARGPARSE FUNCTIONS  #####
##############################################
class DiciphrArgumentParser(argparse.ArgumentParser):
    '''
    A specialized ArgumentParser that adds three options common across all DiCIPHR scripts.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag to ensure options are added only once
        self._options_added = False  

    def _add_common_options(self):
        if self._options_added:
            return
        grp = self.add_argument_group('Other options')
        grp.add_argument(
            '--debug', action='store_true', dest='debug', 
            help='Debug mode: sets logging level, and temporary directories will not be deleted.'
        )
        grp.add_argument(
            '--workdir', action='store', metavar='<dir>', dest='workdir', type=str, default=None,
            help='The directory to preserve intermediate results. If not provided, a temporary directory will be created in $TMPDIR'
        )
        grp.add_argument(
            '--logfile', action='store', metavar='<file>', dest='logfile', type=str, default=None,
            help='A log file. If it does not exist, it will be created. Overrides --logdir'
        )
        grp.add_argument(
            '--logdir', action='store', metavar='<dir>', dest='logdir', type=str, default=None,
            help='A log directory. If it does not exist, it will be created, and a logfile will be created with name generated automatically'
        )
        self._options_added = True

    def parse_args(self, *args, **kwargs):
        self._add_common_options()
        args = super().parse_args(*args, **kwargs)
        # detect workdir 
        if args.workdir is not None:
            make_dir(args.workdir, recursive=True, pass_if_exists=True)
            set_diciphr_tmpdir(args.workdir)
        return args 

    def parse_known_args(self, *args, **kwargs):
        self._add_common_options()
        args, unknown = super().parse_known_args(*args, **kwargs)
        # detect workdir 
        if args.workdir is not None:
            make_dir(args.workdir, recursive=True, pass_if_exists=True)
            set_diciphr_tmpdir(args.workdir)
        return args, unknown

def timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    
class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level

def set_up_logging(filename='', protocol_name='', debug=False, quiet=False, log_basics=False):
    """
    Set up logging to file, stdout (for INFO/DEBUG), and stderr (for WARNING+).
    """
    format_string = "%(asctime)s [{}] [%(levelname)-5.5s]  %(message)s".format(protocol_name or "%(threadName)-12.12s")
    formatter = logging.Formatter(format_string)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear existing handlers
    root_logger.setLevel(logging.DEBUG if debug else (logging.ERROR if quiet else logging.INFO))

    # File handler
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Stdout handler for DEBUG and INFO
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Stderr handler for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)
    
    if debug:
        logging.debug('Logging level set to DEBUG')
    elif quiet:
        logging.warning('Logging level set to ERROR')
    else:
        logging.info('Logging level set to INFO')

    if log_basics:
        logging.info(f'username: {getuser()}')
        logging.info(f'hostname: {gethostname()}')
        logging.info(f'cwd: {os.getcwd()}')
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            logging.info(f'SLURM_JOB_ID: {slurm_job_id}')

def protocol_logging(protocol_name, directory=None, filename=None, debug=False, create_dir=True):
    """
    Set up the log file for a protocol named `protocol_name`.

    Parameters:
        protocol_name (str): Name of the protocol.
        directory (str, optional): Directory where the log file will be created.
        filename (str, optional): Full path to the log file. Overrides `directory` if provided.
        debug (bool, optional): If True, sets logging level to DEBUG and disables cleanup of temp directories.
        create_dir (bool, optional): If True, creates the directory if it doesn't exist.

    Returns:
        str: Path to the log file, or an empty string if no file is used.

    Raises:
        ValueError: If the directory or filename is not writable or invalid.
    """
    log_file = ''

    if filename:
        log_file = filename
        dir_path = os.path.dirname(log_file) or '.'
        if not os.path.exists(dir_path):
            if create_dir:
                os.makedirs(dir_path, exist_ok=True)
            else:
                raise ValueError(f"Directory for log file does not exist: {dir_path}")
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Directory is not writable: {dir_path}")
    elif directory:
        if not os.path.exists(directory):
            if create_dir:
                os.makedirs(directory, exist_ok=True)
            else:
                raise ValueError(f"Directory does not exist: {directory}")
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"Directory is not writable: {directory}")
        _slurm_jid = get_slurm_jobid()
        _timestamp = timestamp()
        log_file = os.path.join(directory, f"{protocol_name}_{_slurm_jid}_{_timestamp}.log")
    set_up_logging(
        filename=log_file,
        protocol_name=protocol_name,
        debug=debug, 
        log_basics=True, 
    )
    return log_file

##############################################
############   MISC FUNCTIONS   ##############
##############################################
def is_flirt_mat_file(filename):
    '''Returns true if filename is a 4x4 text file that satisfies requirements for an affine transformation matrix.'''
    import numpy as np
    try:
        a = np.loadtxt(filename).astype(np.float32)
        if a.shape != (4,4):
            return False
        if all(a[3,:] == [0,0,0,1]):
            return True
    except:
        return False

def is_nifti_file(filename):
    """Returns True if filename is a nifti file. """
    try:
        nib.Nifti1Image.from_filename(filename)
    except:
        return False
    return True
        
def check_inputs(*paths, nifti=False, directory=False, writable=False):
    '''For paths in args, check if inputs exist and are readable, and raise a DiciphrException if not.
    
    Parameters
    ----------
    *paths : str
        Paths to check for existence and readability.
    nifti : bool
        If True, further check that the path is to a valid, readable Nifti file.
    directory : bool
        If True, further check that the path is a readable directory.
    writable : bool
        If True, further check that the path is writable. 
    
    Returns
    -------
    tuple
        Absolute paths of inputs
    '''
    logging.debug('diciphr.utils.check_inputs')
    if len(paths) == 0:
        raise DiciphrException('No files provided')
    
    not_exist=[]
    not_readable=[]
    not_writable=[]
    not_nifti=[]
    not_directory=[]
    for a in paths:
        if not os.access(a, os.F_OK):
            not_exist.append(a)
        elif not os.access(a, os.R_OK):
            not_readable.append(a)
        else:
            if writable and not os.access(a,os.W_OK):
                not_writable.append(a)
            if directory and not os.path.isdir(a):
                not_directory.append(a)
            if nifti and not is_nifti_file(a):
                not_nifti.append(a)
    if not_exist:
        raise DiciphrException('Paths do not exist: {}'.format(','.join(not_exist)))
    if not_readable:
        raise DiciphrException('Paths are not readable: {}'.format(','.join(not_readable)))
    if not_writable:
        raise DiciphrException('Paths are not writable: {}'.format(','.join(not_writable)))
    if not_directory:
        raise DiciphrException('Paths are not directories: {}'.format(','.join(not_directory)))
    if not_nifti:
        raise DiciphrException('Files are not valid nifti: {}'.format(','.join(not_nifti)))
    if len(paths) > 1:
        return tuple(os.path.realpath(_a) for _a in paths)
    else:
        return os.path.realpath(paths[0])

def logical_or(*datas):
    '''
    Get the logical union of some numpy boolean arrays
    Implementation of numpy's logical_or allowing any number of arguments
    '''
    d = datas[0].astype(np.bool)
    for a in datas:
        d = np.logical_or(d, a.astype(np.bool))
    return d

def logical_and(*datas):
    '''
    Get the logical intersection of some numpy boolean arrays
    Implementation of numpy's logical_and allowing any number of arguments
    '''
    d = datas[0].astype(np.bool)
    for a in datas:
        d = np.logical_and(d, a.astype(np.bool))
    return d
    
def force_to_list(arg):
    from collections.abc import Iterable
    if not isinstance(arg, Iterable) or isinstance(arg, str):
        arg = [arg] 
    return list(arg)
    
def is_string(s):
    return isinstance(s, str)

##############################################
############  ENVIRONMENT  ###################
##############################################
def which(program, raise_exception=True):
    """If an executable is at given path or found in PATH environmental variable, return path to command."""
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK) and (not os.path.isdir(fpath))

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"').strip()
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    if raise_exception:
        raise DiciphrException('Cannot find executable: {}'.format(program))
    return None
 
def get_slurm_jobid():
    return os.environ.get('SLURM_JOB_ID', 'interactive')

def get_diciphr_tmpdir():
    return os.environ.get('DICIPHR_TMPDIR', os.environ.get('TMPDIR'))

def set_diciphr_tmpdir(path):
    os.environ['DICIPHR_TMPDIR'] = os.path.realpath(path)
    
##############################################
############   Subprocess   ##################
##############################################
class TerminationSignalReceived(Exception):
    """Custom exception to trigger cleanup on signal."""
    def __init__(self, signum):
        self.signum = signum 
        self.signal_name = signal.Signals(signum).name
        super().__init__(f"Received signal {self.signal_name} ({signum})")
        
    def __str__(self):
        return f"Signal {self.signal_name} ({self.signum}) received"

class ExecCommand:
    """
    A class to execute shell commands in SLURM or HPC environments,
    with support for environment variables, working directories, and real-time logging.

    Attributes:
        cmd_array (list): The command and arguments to execute.
        stdin (str): Optional input to pass to the command.
        environ (dict): Environment variables to set for the command.
        cwd (str): Working directory to run the command in.
        quiet (bool): If True, suppress logging output.
    """

    def __init__(self, cmd_array, stdin=None, environ=None, cwd=None, quiet=False):
        if not isinstance(cmd_array, (list, tuple)):
            raise TypeError("cmd_array must be a list or tuple of command arguments")
        self.cmd_array = list(map(str, cmd_array))
        self.stdin = stdin
        self.environ = environ or {}
        self.cwd = cwd
        self.quiet = quiet
        self.process = None 
        # Register SIGTERM and SIGINT handlers
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        signal.signal(signal.SIGINT, self._handle_termination_signal)
        signal.signal(signal.SIGXCPU, self._handle_termination_signal)
        signal.signal(signal.SIGUSR1, self._handle_termination_signal)
    
    def _handle_termination_signal(self, signum, frame):
        raise TerminationSignalReceived(signum)
    
    def _flush_logs(self):
        for handler in logging.root.handlers:
            try:
                handler.flush()
            except Exception as e:
                print(f"[WARN] Failed to flush log handler: {e}", file=sys.stderr)
                
    def run(self, raise_on_error=True):
        env = os.environ.copy()
        env.update(self.environ)
        logging.info("Run command:")
        logging.info(" ".join(self.cmd_array))
        if self.environ:
            logging.info(f"Provided environmental variables: {self.environ}")
        try:
            self.process = subprocess.Popen(
                self.cmd_array,
                stdin=subprocess.PIPE if self.stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env,
                universal_newlines=True,
                bufsize=1  # Line-buffered
            )

            def log_stream(stream, log_func):
                for line in iter(stream.readline, ''):
                    log_func(line.rstrip())
                stream.close()

            threads = []
            if not self.quiet:
                threads.append(threading.Thread(target=log_stream, args=(self.process.stdout, logging.info)))
                threads.append(threading.Thread(target=log_stream, args=(self.process.stderr, logging.warning)))

            for t in threads:
                t.start()

            if self.stdin:
                self.process.stdin.write(self.stdin)
                self.process.stdin.close()

            for t in threads:
                t.join()

            returncode = self.process.wait()

            if returncode != 0:
                if raise_on_error:
                    raise subprocess.CalledProcessError(returncode, self.cmd_array)
                else:
                    logging.warning(f"Command failed with return code {returncode}")

            self._flush_logs()
            return returncode

        except subprocess.CalledProcessError as e:
            logging.exception(f"Command failed with return code {e.returncode}")
            return e.returncode
        
        except TerminationSignalReceived as e:
            logging.warning(f"Terminating due to signal: {e.signal_name}")
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    logging.info(f"Subprocess terminated cleanly after signal {e.signal_name}.")
                except Exception as ex:
                    logging.error(f"Failed to terminate subprocess: {ex}")
            self._flush_logs()
            raise 
        
        except Exception as e:
            logging.exception("Command execution failed")
            raise RuntimeError("Command execution failed") from e

class ExecFSLCommand(ExecCommand):
    """
    Special case of ExecCommand for FSL programs
    If the system has FSLSIF environmental set, will use apptainer
    """
    
    def __init__(self, cmd_array, stdin=None, environ=None, cwd=None, quiet=False, 
                gpu=False, container_path=None, container_fslbin="/usr/local/fsl/bin"):
        if not isinstance(cmd_array, (list, tuple)):
            raise TypeError("cmd_array must be a list or tuple of command arguments")
        cmd_array = list(map(str, cmd_array))
        if container_path:
            fslsif = container_path
        else:
            fslsif = os.environ.get('FSLSIF',None)
        if fslsif:
            fslcmd = os.path.basename(cmd_array[0])
            # use apptainer 
            prepend_cmd = ["apptainer","exec"] 
            if gpu:
                prepend_cmd.extend(["--nv"])
            env_string = "FSLOUTPUTTYPE=NIFTI_GZ"
            if environ:
                for k, v in environ.items():
                    env_string += f",{k}={v}"
            prepend_cmd.extend(["--cleanenv","--env",env_string])
            _diciphr_tmpdir = get_diciphr_tmpdir()
            if _diciphr_tmpdir:
                prepend_cmd.extend(["--bind",f"{_diciphr_tmpdir}:{_diciphr_tmpdir}"])
            prepend_cmd.append(fslsif)
            # special case for eddy 
            if fslcmd == 'eddy':
                prepend_cmd.extend([f"{container_fslbin}/fslpython", f"{container_fslbin}/{fslcmd}"])
            else:
                prepend_cmd.extend([f"{container_fslbin}/{fslcmd}"])
            cmd_array = prepend_cmd + cmd_array[1:]
        super().__init__(cmd_array, stdin=stdin, environ=environ, cwd=cwd, quiet=quiet)
