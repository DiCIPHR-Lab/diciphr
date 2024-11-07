# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:04:32 2016

@author: parkerwi
"""

import os, sys, logging, tempfile, shutil, subprocess, datetime, select, time
import nibabel as nib
import numpy as np
from socket import gethostname
from getpass import getuser

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
    
def make_dir(path, recursive=False, pass_if_exists=False, verbose=False):
    '''
    Make a directory, optionally recursive, and return path
    '''
    if not os.path.isdir(path):
        if recursive:
            mkdir_func=os.makedirs
        else:
            mkdir_func=os.mkdir
        try:
            mkdir_func(path)
        except Exception as e:
            if pass_if_exists:
                logging.warning(str(e))
            else:
                raise e 
    if verbose:
        logging.info("Made directory {}".format(path))
    return path

def make_temp_dir(directory=None, prefix='tmp'):
    '''Makes and returns a path to a unique working directory.
    
    If directory is None,
        if environmental variable CBICA_TMPDIR exists, use that
        else, make folder in python module tmpfile's tmp directory/tmp
    else,
        make a new folder inside directory
        
    Parameters 
    ----------
    directory : Optional[str]
        Directory inside of which to create a new tempdir
    prefix : Optional[str]
        Prefix of the new tempdir, default 'tmp'
        
    Returns
    -------
    str
        The newly created tempdir.
    '''
    from uuid import uuid4    
        
    if directory is None:
        directory = os.environ.get('TMPDIR',None)
    
    try:
        workdir = tempfile.mkdtemp(prefix=prefix, dir=directory, suffix=str(uuid4()))
    except Exception:
        workdir = tempfile.mkdtemp(prefix=prefix, dir=directory)
    logging.info('Created temporary directory {}'.format(workdir))
    return workdir
    
def random_string(length=16):
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
    
##############################################
############   MISC FUNCTIONS   ##############
##############################################

def set_up_logging(filename='', protocol_name='', tee_to_stderr=True, debug=False, quiet=False):
    '''
    Convenience function to set up logging environment.
    
    Parameters 
    ----------
    filename : Optional[str]
        The log filename 
    protocol_name : Optional[str]
        A name to identify the script e.g. MyScript
        Will default logging module's threadName 
    tee_to_stderr : Optional[bool]
        If true, logging messages will also be printed to stderr. Default true. 
    debug : Optional[bool]
        If true, sets logging level to DEBUG. Default true. 
    quiet : Optional[bool]
        If true, sets logging level to ERROR. Default true. 
    '''
    
    format_string = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    if protocol_name:    
        format_string = "%(asctime)s ["+protocol_name+" ] [%(levelname)-5.5s]  %(message)s"
    logFormatter = logging.Formatter(format_string)
    rootLogger = logging.getLogger()
    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
    if tee_to_stderr:
        #http://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
def log_basics():
    '''Log some basic info about a SGE job for pipelining.'''
    logging.info('username: {}'.format(getuser()))
    logging.info('hostname: {}'.format(gethostname()))
    logging.info('cwd: {}'.format(os.getcwd()))
    try:
        logging.info('jobid: {}'.format(os.environ.get('JOB_ID')))
        logging.info('jobname: {}'.format(os.environ.get('JOB_NAME')))
    except:
        pass
    
def protocol_logging(protocol_name, path=None, debug=False):
    '''Set up the log file for a protocol named protocol_name at a path. 
    Path can be file or directory (a file will be created).
    If path is None or empty string, logging will only print to stderr.'''
    timestamp=time.strftime('%Y%m%d-%H%M%S.%s')
    if path is None:
        log_file=''
    else:
        if os.path.isdir(path):
            log_file = os.path.join(path,'{}_{}.log'.format(protocol_name,timestamp))
        else:
            log_file = path
    set_up_logging(filename=log_file,protocol_name=protocol_name,tee_to_stderr=True,debug=debug)
    log_basics()
    return log_file
    
def is_nifti_file(filename):
    """Returns True if filename is a nifti file. """
    try:
        im = nib.Nifti1Image.from_filename(filename)
    except:
        return False
    return True

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
        
def check_inputs(*args, **kwargs):
    '''For paths in args, check if inputs exist and are readable, and raise a DiciphrException if not.
    
    Parameters
    ----------
    *args : str
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
    if len(args) == 0:
        raise DiciphrException('No files provided')
    
    nifti = kwargs.get('nifti', False)
    directory = kwargs.get('directory', False)
    writable = kwargs.get('writable', False)
    
    not_exist=[]
    not_readable=[]
    not_writable=[]
    not_nifti=[]
    not_directory=[]
    for a in args:
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
    if len(args) > 1:
        return list(os.path.realpath(_a) for _a in args)
    else:
        return os.path.realpath(args[0])

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
        
##############################################
############   CLASSES   #####################
##############################################        
class ExecCommand(object):
    '''
    ExecCommand object
    
    __init__
    set_environment - changes environment
    run -- runs and returns returncode,stderr file-like object stdout file-like object
    pop_stderr
    pop_stdout
    get_stderr get a pointer to file-like object
    get_stdout get a pointer to file-like object
    print_stderr = prints and removes file-like object
    print_stdout = prints and removes file-like object
    __del__ = prints to stderr/stdout if it still exists
    '''
    
    def __init__(self,cmd_array,stdin=None,environ={},cwd=None,quiet=False):
        self.stdin = stdin
        self.cmd_array = [str(a) for a in cmd_array]
        self.environ = os.environ.copy()
        if environ:
            for k,v in environ.items():
                self.environ[k] = v
                logging.debug('Updating environmental variable {0} to {1}'.format(k,v))
        self.workdir = cwd
        self.log_note = ''
        self.quiet = quiet
        
    def set_stdin(self,openfile):
        self.stdin = openfile

    def make_temp_dir(self):
        self.workdir = make_temp_dir(prefix=self.cmd_array[0][:8])
                
    def set_environment(self,environ):
        self.environ = os.environ.copy()
        for k,v in environ.items():
            self.environ[k]=v
            
    def set_up(self):
        stdin=None if self.stdin is None else subprocess.PIPE
        self._process = subprocess.Popen(
                    ' '.join(self.cmd_array),
                    stdin=self.stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self.environ,
                    cwd=self.workdir,
                    shell=True
        )
        self.stdout = self._process.stdout
        self.stderr = self._process.stderr
        if self.stdin is not None:
            self.log_note += 'from PIPE'
            self.stdin.close()
        self.poll = self._process.poll
        
    def communicate(self):
        if self.stdin is not None:
            self.stdin.close()
        ret = self._process.communicate()
        self.returncode = self._process.returncode
        return ret        
        
    def wait(self):
        return self._process.wait()
        
    def run(self):
        self.returncode = -1 
        if self.workdir:
            logging.info('Executing command [ {} ] in dir {}'.format(' '.join(self.cmd_array),self.workdir))
        else:
            logging.info('Executing command [ {} ]'.format(' '.join(self.cmd_array)))
        try:
            self.set_up()
            ret = [None,'','']
            while True:
                reads = [self.stdout.fileno(), self.stderr.fileno()]
                R = select.select(reads, [], [])
                for fd in R[0]:
                    if fd == self.stderr.fileno():
                        line = self.stderr.readline()
                        try:
                            line = str(line.decode('utf-8','replace')).strip()
                        except AttributeError:
                            line = str(line).strip()
                        if ( not self.quiet ) and bool(line): 
                            logging.warning(line)
                        ret[2] += line
                    if fd == self.stdout.fileno():
                        line = self.stdout.readline()
                        try:
                            line = str(line.decode('utf-8','replace')).strip()
                        except AttributeError:
                            line = str(line).strip()
                        if ( not self.quiet ) and bool(line): 
                            logging.info(line)
                        ret[1] += line
                if self.poll() is not None:
                    break
            ret[0] = self.poll()
            self.returncode = ret[0]
        except Exception as e:
            logging.error(repr(e))
            raise e
        finally:
            if self.returncode != 0:
                if self.log_note:
                    self.log_note = 'return code {}:'.format(self.returncode) + self.log_note 
                else:
                    self.log_note = 'return code {}'.format(self.returncode)
                logging.error(self.log_note)
        self.stdout = ret[1]
        self.stderr = ret[2]
        return ret
        
    def pipe_to(self,ExecCommandInstance):
        self.set_up()
        ExecCommandInstance.set_stdin(self.stdout)
        #ExecCommandInstance.set_up()
        #ExecCommandInstance.stdin.close()
                
    def get_stdout(self):
        return self.stdout
        
    def get_stderr(self):
        return self.stderr

def force_to_list(arg):
    from collections import Iterable
    try:
        # python 2.x
        if not isinstance(arg, Iterable) or isinstance(arg, basestring):
            arg = [arg] 
    except:
        # python 3.x
        if not isinstance(arg, Iterable) or isinstance(arg, str):
            arg = [arg] 
    return list(arg)
    
def is_string(s):
    try:
        #python 2 
        return isinstance(s, basestring)
    except NameError:
        #python 3 
        return isinstance(s, str)    
        
def qsub_submit(command, args=[], queue=None, job_name=None, hold_jid=[]):
    cmd=['qsub']
    if is_string(command):
        command = [command]
    if queue is not None:
        cmd.extend(['-l',queue])
    if job_name is not None:
        cmd.extend(['-N',job_name])
    if hold_jid:
        if is_string(hold_jid):
            hold_jid=[hold_jid]
        cmd.extend(['-hold_jid',','.join(hold_jid)])
    cmd.extend(command)
    returncode, stdout, stderr = ExecCommand(cmd).run()
    job_id = stdout.split(' ')[2]
    return job_id
