# (c) 2007 The Board of Trustees of the University of Illinois.

import sys
import os
from os import path
import re
from itertools import imap, repeat, chain

import globals
import process
from futures import Future

class Benchmark(object):
    """A benchmark.

    If the benchmark is malformed or otherwise invalid, only the 'name' and
    'invalid' fields will be set.  Otherwise all fields will be set.

    Fields:
      name        The name of the benchmark.  This is also the benchmark
                  directory name.
      invalid     None if the benchmark is valid; otherwise, an exception
                  describing why the benchmark is invalid.
      path        Full path of the benchmark directory.
      descr       A description of the benchmark.
      impls       A dictionary of benchmark source implementations.
      datas       A dictionary of data sets used to run the benchmark."""

    def __init__(self, name, path = None, impls = [], datasets = [],
                 description=None, invalid=None):
        self.name = name
        self.invalid = invalid

        if invalid is None:
            self.path = path
            self.impls = dict(imap(lambda i: (i.name, i), impls))
            self.datas = dict(imap(lambda i: (i.name, i), datasets))
            self.descr = description

    def createFromName(name):
        """Scan the benchmark directory for the benchmark named 'name'
        and create a benchmark object for it."""
        bmkpath = path.join(globals.root, 'benchmarks', name)
        descr = process.read_description_file(bmkpath)

        try:
            # Scan implementations of the benchmark
            impls = [BenchImpl.createFromName(name, impl)
                     for impl in process.scan_for_benchmark_versions(bmkpath)]
            
            # Scan data sets of the benchmark
            datas = [BenchDataset.createFromName(name, data)
                     for data in process.scan_for_benchmark_datasets(bmkpath)]

            # If no exception occurred, the benchmark is valid
            return Benchmark(name, bmkpath, impls, datas, descr)
        except Exception, e:
            return Benchmark(name, invalid=e)
        
    createFromName = staticmethod(createFromName)

    def describe(self):
        """Return a string describing this benchmark."""

        if self.invalid:
            return "Error in benchmark:\n" + str(self.invalid)

        if self.descr  is None:
            header = "Benchmark '" + self.name + "'"
        else:
            header = self.descr

        impls = " ".join([impl.name for impl in self.impls.itervalues()])
        datas = " ".join([data.name for data in self.datas.itervalues()])

        return header + "\nVersions: " + impls + "\nData sets: " + datas

    def instance_check(x):
        if not isinstance(x, Benchmark):
            raise TypeError, "argument must be an instance of Benchmark"

    instance_check = staticmethod(instance_check)

class BenchImpl(object):
    """An implementation of a benchmark."""

    def __init__(self, name, description=None):
        if not isinstance(name, str):
            raise TypeError, "name must be a string"

        self.name = name
        self.descr = description

    def createFromName(name, impl):
        """Scan the directory containing a benchmark implementation
        and create a BenchImpl object from it."""

        # Path to the implementation
        impl_path = path.join(globals.root, 'benchmarks', name, 'src', impl)

        # Get the description from a file, if provided
        descr = process.read_description_file(impl_path)

        return BenchImpl(impl, descr)

    createFromName = staticmethod(createFromName)

    def makefile(self, benchmark, target=None, action=None):
        """Run this implementation's makefile."""
        
        Benchmark.instance_check(benchmark)

        def perform():
            srcdir = path.join('src', self.name)
            builddir = path.join('build', self.name)

            env={'SRCDIR':srcdir,
                 'BUILDDIR':builddir,
                 'BIN':path.join(builddir,benchmark.name),
                 'PARBOIL_ROOT':globals.root}

            # Run the makefile to build the benchmark
            return process.makefile(target=target,
				    action=action,
                                    filepath=path.join(srcdir, "Makefile"),
                                    env=env)

        # Go to the benchmark directory before building
        return process.with_path(benchmark.path, perform)

    def build(self, benchmark):
        """Build an executable of this benchmark implementation."""
        return self.makefile(benchmark)

    def isBuilt(self, benchmark):
        """Determine whether the executable is up to date."""
        return self.makefile(benchmark, action='q')

    def clean(self, benchmark):
        """Remove build files for this benchmark implementation."""
        return self.makefile(benchmark, 'clean')

    def run(self, benchmark, dataset, do_output=True, extra_opts=[]):
        """Run this benchmark implementation.

        Return True if the benchmark terminated normally or False
        if there was an error."""

        # Ensure that the benchmark has been built
        if not self.isBuilt(benchmark):
            rc = self.build(benchmark)

            # Stop if 'make' failed
            if not rc: return False

        def perform():
            # Run the program
            exename = path.join('build', self.name, benchmark.name)
            args = [exename] + extra_opts + dataset.getCommandLineArguments(do_output)
            rc = process.spawnwaitv(exename, args)

            # Program exited with error?
            if rc != 0: return False
            return True

        return process.with_path(benchmark.path, perform)

    def check(self, benchmark, dataset):
        """Check the output from the last run of this benchmark
        implementation.

        Return True if the output checks successfully or False
        otherwise."""

        def perform():
            output_file = dataset.getTemporaryOutputPath()
            reference_file = dataset.getReferenceOutputPath()

            compare = os.path.join('tools', 'compare-output')
            rc = process.spawnwaitl(compare,
                                    compare, reference_file, output_file)

            # Program exited with error, or mismatch in output?
            if rc != 0: return False
            return True

        return process.with_path(benchmark.path, perform)

    def __str__(self):
        return "<BenchImpl '" + self.name + "'>"

class BenchDataset(object):
    """Data sets for running a benchmark."""

    def __init__(self, name, in_files=[], out_files=[], parameters=[],
                 description=None):
        if not isinstance(name, str):
            raise TypeError, "name must be a string"

        self.name = name
        self.inFiles = in_files
        self.outFiles = out_files
        self.parameters = parameters
        self.descr = description

    def createFromName(name, dset):
        """Scan the directory containing a dataset
        and create a BenchDataset object from it."""

        # Identify the paths where files may be found
        benchmark_path = path.join(globals.root, 'benchmarks', name)

        if path.exists(path.join(benchmark_path, 'input')):
            input_path = path.join(benchmark_path, 'input', dset)
        else:
            input_path = None
    
        output_path = path.join(benchmark_path, 'output', dset)

        # Look for input files
        
        def check_default_input_files():
            # This function is called to see if the input file set
            # guessed by scanning the input directory can be used
            if invalid_default_input_files:
                raise ValueError, "Cannot infer command line when there are multiple input files in a data set\n(Fix by adding an input DESCRIPTION file)"
                
        if input_path:
            input_descr = process.read_description_file(input_path)
            input_files = list(process.scan_for_files(input_path,
                                                      boring=['DESCRIPTION','.svn']))

            # If more than one input file was found, cannot use the default
            # input file list produced by scanning the directory
            invalid_default_input_files = len(input_files) > 1
        else:
            # If there's no input directory, assume the benchmark
            # takes no input
            input_descr = None
            input_files = []
            invalid_default_input_files = False

        # Read the text of the input description file
        if input_descr is not None:
            (parameters, input_files1, input_descr) = \
                unpack_dataset_description(input_descr, input_files=None)

            if input_files1 is None:
                # No override vaule given; use the default
                check_default_input_files()
            else:
                input_files = input_files1
        else:
            check_default_input_files()
            parameters = []

        # Look for output files
        output_descr = process.read_description_file(output_path)
        output_files = list(process.scan_for_files(output_path,
                                                   boring=['DESCRIPTION','.svn']))
        if len(output_files) > 1:
            raise ValueError, "Multiple output files not supported"

        # Concatenate input and output descriptions
        if input_descr and output_descr:
            descr = input_descr + "\n\n" + output_descr
        else:
            descr = input_descr or output_descr

        return BenchDataset(dset, input_files, output_files, parameters, descr)

    createFromName = staticmethod(createFromName)

    def getTemporaryOutputPath(self):
        """Get the name of a file used to hold the output of a benchmark run.
        This function should always return the same name if its parameters
        are the same.  The output path is not the path where the reference
        output is stored."""

        return path.join('run', self.name, self.outFiles[0])

    def getReferenceOutputPath(self):
        """Get the name of the reference file, to which the output of a
        benchmark run should be compared."""

        return path.join('output', self.name, self.outFiles[0])

    def getCommandLineArguments(self, do_output=True):
        """Get the command line arguments that should be passed to the
        executable to run this data set.  If 'output' is True, then
        the executable will be passed flags to save its output to a file.

        Directories to hold ouptut files are created if they do not exist."""
        args = []

        # Add arguments to pass input files to the benchmark
        if self.inFiles:
            in_files = ",".join([path.join('input', self.name, x)
                                 for x in self.inFiles])
            args.append("-i")
            args.append(in_files)

        # Add arguments to store the output somewhere, if output is
        # desired
        if do_output and self.outFiles:
            if len(self.outFiles) != 1:
                raise ValueError, "only one output file is supported"

            out_path = self.getTemporaryOutputPath()
            args.append("-o")
            args.append(out_path)

            # Ensure that a directory exists for the output
            process.touch_directory(path.dirname(out_path))

        args += self.parameters
        return args

    def __str__(self):
        return "<BenchData '" + self.name + "'>"

def unpack_dataset_description(descr, parameters=[], input_files=[]):
    """Read information from the raw contents of a data set description
    file.  Optional 'parameters' and 'input_files' arguments may be
    given, which will be retained unless overridden by the description
    file."""
    leftover = []
    split_at_colon = re.compile(r"^\s*([a-zA-Z]+)\s*:(.*)$")

    # Initialize these to default empty strings
    parameter_text = None
    input_file_text = None
    
    # Scan the description line by line
    for line in descr.split('\n'):
        m = split_at_colon.match(line)
        if m is None: continue

        # This line appears to declare something that should be
        # interpreted
        keyword = m.group(1)
        if keyword == "Parameters":
            parameter_text = m.group(2)
        elif keyword == "Inputs":
            input_file_text = m.group(2)
        # else, ignore the line

    # Split the strings into (possibly) multiple arguments, discarding
    # whitespace
    if parameter_text is not None: parameters = parameter_text.split()
    if input_file_text is not None: input_files = input_file_text.split()
    return (parameters, input_files, descr)

def find_benchmarks():
    """Find benchmarks in the repository.  The benchmarks are
    identified, but their contents are not scanned immediately.  A
    dictionary is returned mapping benchmark names to futures
    containing the benchmarks."""

    if not globals.root:
        raise ValueError, "root directory has not been set"

    # Scan all benchmarks in the 'benchmarks' directory and
    # lazily create benchmark objects.
    db = {}
    try:
        for bmkname in process.scan_for_benchmarks(globals.root):
            bmk = Future(lambda bmkname=bmkname: Benchmark.createFromName(bmkname))
            db[bmkname] = bmk
    except OSError, e:
        sys.stdout.write("Benchmark directory not found!\n\n")
        return {}

    return db
