# (c) 2007 The Board of Trustees of the University of Illinois.

# These are the main actions that the driver may take.  The actions
# call lower-level routines from the process or benchmark modules.

import os
from itertools import imap

import process
import benchmark
import globals
from text import format_columns

def benchmark_iter():
    """Iterate over the benchmarks in 'bmks'."""
    # bmks is a dictionary from str to Future(Benchmark)
    return imap(lambda x: x.get(), globals.benchmarks.itervalues())

def list_benchmarks():
    """List all benchmarks on standard output."""
    print "Benchmarks:"
    for bmk in benchmark_iter(): print "  " + bmk.name
        
def describe_benchmarks():
    """Print descriptions of all benchmarks to standard output."""
    for bmk in benchmark_iter(): describe_benchmark(bmk)

def describe_benchmark(bmk):
    """Print a description of one benchmark to standard output."""

    print "  " + bmk.name
    print format_columns(bmk.describe(), 4)

def lookup_benchmark(name):
    """Find a benchmark, given its name.  Returns None if no benchmark
    is found with the given name or if the benchmark is invalid.
    If the benchmark cannot be found, an error is printed."""
    b = globals.benchmarks.get(name)
    if b is not None:
        bmk = b.get()
        if bmk.invalid is None:
            return bmk
        else:
            print "Error in benchmark:"
            print str(bmk.invalid)
            return None
    else:
        print "Cannot find benchmark"
        return None

def with_benchmark_named(name, action):
    """Look up the benchmark named 'name'.  If found, apply the action
    to it.  Otherwise, print an error message and return None."""

    bmk = lookup_benchmark(name)
    if bmk is not None: action(bmk)

def compile_benchmark(bmk, version_name):
    """Compile the benchmark 'bmk'."""
    try: impl = bmk.impls[version_name]
    except KeyError:
        print "Cannot find benchmark version"
        return
    
    impl.build(bmk)

def clean_benchmark(bmk, version_name=None):
    """Remove the compiled code for one implementation of 'bmk'.  If
    no version is given, clean all versions."""

    if version_name:
        try: impl = bmk.impls[version_name]
        except KeyError:
            print "Cannot find benchmark version"
            return

        impl.clean(bmk)
    else:
        # Clean all versions
        for impl in bmk.impls.itervalues():
            impl.clean(bmk)

def run_benchmark(bmk, version_name, input_name, check=True, extra_opts=[]):
    """Run the benchmark 'bmk'."""
    try: impl = bmk.impls[version_name]
    except KeyError:
        print "Cannot find benchmark version"
        return
    
    try: data = bmk.datas[input_name]
    except KeyError:
        print "Cannot find data set"
        return

    # Run the benchmark
    success = impl.run(bmk, data, check, extra_opts=extra_opts)

    if not success:
        print "Run failed!"
        return

    # Verify the output
    if check:
        success = impl.check(bmk, data)

        if not success:
            print "Output checking tool detected a mismatch"
    else:
        print "Output was not checked for correctness"
