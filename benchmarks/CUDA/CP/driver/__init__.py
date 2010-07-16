# (c) 2007 The Board of Trustees of the University of Illinois.

import sys
import os
from itertools import imap

import globals
import actions
import options

def run():
    # Print a banner message
    print "Parboil parallel benchmark suite, version 0.1"
    print
    
    # Global variable setup
    root_path = os.getcwd()
    python_path = (os.path.join(root_path,'common','python') +
                   ":" +
                   os.environ.get('PYTHONPATH',""))
    
    globals.root = root_path
    globals.benchmarks = benchmark.find_benchmarks()
    globals.program_env = {'PARBOIL_ROOT':root_path,
                           'PYTHONPATH':python_path,
                           }

    # Parse options
    act = options.parse_options(sys.argv)

    # Perform the specified action
    if act: act()

