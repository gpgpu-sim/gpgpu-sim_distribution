# (c) 2007 The Board of Trustees of the University of Illinois.

# Process-management and directory management routines are collected here.

import os
import os.path as path
import stat
from itertools import imap, ifilter, chain

import globals

def scan_for_files(topdir, directory=False, boring=[]):
    """Scan the contents of the directory 'topdir'.  If 'directory' is
    True, return a sequence of all directories found in that directory;
    otherwise, return a sequence of all files found in that directory.
    Directories whose names are found in the 'boring' list are excluded."""

    # Look for directories or regular files, depending on the 'directory'
    # parameter
    if directory: valid_test = path.isdir
    else: valid_test = path.isfile
    
    def interesting(fname):
        # True if 'dirname' is not a boring name, and it is a directory
        if fname in boring: return False
        
        fpath = path.join(topdir, fname)
        try: return valid_test(fpath)
        except OSError: return False    # Ignore file-not-found errors

    if not path.isdir(topdir):
        raise OSError, "Cannot access '" + str(topdir) + "' as a directory"

    # Return names of all directories found
    return ifilter(interesting, os.listdir(topdir))

def scan_for_benchmarks(topdir):
    """Scan subdirectories of the benchmark repository to find
    benchmarks.  Return a sequence containing all benchmark
    directory names."""

    return scan_for_files(path.join(topdir, "benchmarks"), True, boring=['_darcs','.svn'])
        
def scan_for_benchmark_versions(bmkdir):
    """Scan subdirectories of a benchmark directory 'bmkdir' to find
    benchmark versions.  Return a sequence containing all benchmark
    version names."""

    return scan_for_files(path.join(bmkdir, "src"), True, boring=['.svn'])

def scan_for_benchmark_datasets(bmkdir):
    """Scan subdirectories of a benchmark directory 'bmkdir' to find
    data sets.  Return a sequence containing all data set names."""

    # Get input and output subdirectories
    inp_dir = path.join(bmkdir, "input")
    if path.exists(inp_dir):
        inp_dirs = scan_for_files(path.join(bmkdir, "input"), True, boring=['.svn'])
    else:
        # Assume no input files are found because the benchmark
        # doesn't need input
        inp_dirs = []
        
    out_dirs = scan_for_files(path.join(bmkdir, "output"), True, boring=['.svn'])

    # Return the union of the dataset directories
    return dict(imap(lambda x: (x, None), chain(inp_dirs, out_dirs))).keys()

def read_description_file(dirpath):
    """Read the contents of a file in 'dirpath' called DESCRIPTION,
    if one exists.  This returns the file text as a string, or None
    if no description was found."""

    descr_path = os.path.join(dirpath, 'DESCRIPTION')
    if os.access(descr_path, os.R_OK):
        descr_file = file(descr_path, 'r')
        descr = descr_file.read()
        descr_file.close()
        return descr
    
    # else, return None

def touch_directory(dirpath):
    """Ensures that the directory 'dirpath' and its parent directories
    exist.  If they do not exist, they will be created.  It is an
    error if the path exists but is not a directory."""
    if path.isdir(dirpath):
        return
    elif path.exists(dirpath):
        raise OSError, "Path exists but is not a directory"
    else:
        (head, tail) = path.split(dirpath)
        if head: touch_directory(head)
        os.mkdir(dirpath)

def with_path(wd, action):
    """Executes an action in a separate working directory.  The action
    should be a callable object."""
    cwd = os.getcwd()
    os.chdir(wd)
    try: result = action()
    finally: os.chdir(cwd)
    return result
    
def makefile(target=None, action=None, filepath=None, env={}):
    """Run a makefile.  An optional command, makefile path, and dictionary of
    variables to define on the command line may be defined.  The return code
    value is the return code returned by the makefile.

    If no action is given, 'make' is invoked.  Returns True if make was
    successful and False otherwise.

    A 'q' action queries whether the target needs to be rebuilt.  True is
    returned if the target is up to date."""

    args = ["make"]

    if action is None:
        def run():
            rc = os.spawnvp(os.P_WAIT, "make", args)
            return rc == 0
    elif action in ['q']:
        args.append('-q')

        def run():
            rc = os.spawnvp(os.P_WAIT, "make", args)
            if rc == 0:
                # Up-to-date
                return True
            elif rc == 1:
                # Needs remake
                return False
            else:
                # Error
                return False
    else:
        raise ValueError, "invalid action"

    # Pass the target as the second argument
    if target: args.append(target)

    # Pass the path the the makefile
    if filepath:
        args.append('-f')
        args.append(filepath)

    # Pass variables
    for (k,v) in env.iteritems():
        args.append(k + "=" + v)

    # Print a status message, if running in verbose mode
    if globals.verbose:

        print "Running '" + " ".join(args) + "' in " + os.getcwd()

    # Run the makefile and return result info
    return run()

def spawnwaitv(prog, args):
    """Spawn a program and wait for it to complete.  The program is
    spawned in a modified environment."""

    env = dict(os.environ)
    env.update(globals.program_env)

    # Print a status message if running in verbose mode
    if globals.verbose:
        print "Running '" + " ".join(args) + "' in " + os.getcwd()

    # Check that the program is runnable
    if not os.access(prog, os.X_OK):
        raise OSError, "Cannot execute '" + prog + "'"

    # Run the program
    return os.spawnve(os.P_WAIT, prog, args, env)

def spawnwaitl(prog, *argl):
    """Spawn a program and wait for it to complete.  The program is
    spawned in a modified environment."""

    return spawnwaitv(prog, argl)
