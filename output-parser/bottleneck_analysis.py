#import the necessary libraries
from os import error
import matplotlib.pyplot as plt
import re
import sys
import linecache

#checks for correct input format
if (len(sys.argv)!=2): 
    sys.exit("The format is 'python3 bottleneck_analysis.py <filename>'")
filename=sys.argv[1]

"""
The Basic idea here is to identfy the three most expensive kernels in your program
and then give you a rough idea where the basic bottle-neck is. This tool is based on
a part of the documentation for GP-GPUsim (MICRO 2012). Link to the above mentioned: http://gpgpu-sim.org/manual/index.php/Main_Page#:~:text=gpu_tot_sim_insn%20/%20wall_time-,Simple%20Bottleneck%20Analysis,-These%20performance%20counters

The following is the implimentation theme of the tool:
1. Make a dictionary with each thread launch's kernel_uid as key and line number as value.
2. Take out and all the necessary counters for bottle-neck analysis.
3. Subtract from the preceeding occurence as these are incremental, not kernel specific.
4. Group all the uids with same name together.
5. Get TOP-3 expensive kernels.
6. Plot the counters on the graph using plt for each kernel.

"""


#this function gets the uid and line for each kernel
def uid_line(file):
    uid_pattern=re.compile("kernel_launch_uid = (\d+)")
    res={}
    for i,line in enumerate(open(file)):
        for match in re.finditer(uid_pattern,line):
            capture_id=list(match.group(1))
            capture_id=int(''.join(capture_id))
            if capture_id not in res:
                res[capture_id]={}
            res[capture_id]["line"]=i
            res[capture_id]["uid"]=capture_id
    
    return res

#this kernel gets a particular figure from a particular kernel, given its starting line
def fetch_figure(fp,stat,kernel_line):
    line_no=kernel_line
    pattern=re.compile("^"+stat+" = ([+-]?[0-9]+\.?[0-9]*|\.[0-9]+)")
    end_ker_pattern=re.compile("^.*END-of-Interconnect-DETAILS.*$")

    matcher=re.match(pattern,linecache.getline(fp,line_no))
    while(not bool(matcher)):
        if(bool(re.match(end_ker_pattern,linecache.getline(fp,line_no)))):
            raise Exception("There is no such metric, please check the white spaces and perform a spell check")
        line_no+=1
        matcher=re.match(pattern,linecache.getline(fp,line_no))

    figure=list(matcher.group(1))
    figure=float(''.join(figure))
    return figure


