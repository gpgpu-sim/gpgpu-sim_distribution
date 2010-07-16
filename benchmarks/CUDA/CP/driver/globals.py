# (c) 2007 The Board of Trustees of the University of Illinois.

# This file holds global variables used by the driver.

# Root directory of the repository (str)
root = None

# Benchmarks in the repository ({str:Future(Benchmark)})
benchmarks = None

# Environment variables to use when spawning subprograms ({str:str})
program_env = None

# True if verbose output is desired.  This may be set during
# option parsing.
verbose = False
