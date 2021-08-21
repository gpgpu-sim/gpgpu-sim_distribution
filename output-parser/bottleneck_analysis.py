#import the necessary libraries
import matplotlib.pyplot as plt
import re
import sys
import linecache

#checks for correct input format
if (len(sys.argv)!=2): 
    sys.exit("The format is 'python3 analyser.py <filename>'")

filename=sys.argv[1]

"""
The Basic idea here is to identfy the three most expensive kernels in your program
and then give you a rough idea where the basic bottle-neck is. This tool is based on
a part of the documentation for GP-GPUsim (MICRO 2012). Link to the above mentioned: http://gpgpu-sim.org/manual/index.php/Main_Page#:~:text=gpu_tot_sim_insn%20/%20wall_time-,Simple%20Bottleneck%20Analysis,-These%20performance%20counters

The following is the implimentation theme of the tool:


"""