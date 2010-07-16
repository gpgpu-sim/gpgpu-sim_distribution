#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Cole Trapnell on 2007-05-01.
Copyright (c) 2007 __MyCompanyName__. All rights reserved.
"""

import sys
import getopt
import os
import time
import timeit
import csv

help_message = '''
This script runs the full cmatch experimental suite.
'''


class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg

#refs = ["NC_005810", "2","NC_003997"]
#qry_lengths = [25, 50, 200, 800]

refs = ["NC_003997"]
qry_lengths = [25]

def get_stats(filename):
	stats = {}
	statfile = open(filename)
	stats = dict([(key, float(value)) for (key, value) in csv.reader(statfile)])
	return stats

def cmatch_query_string(ref, length):
	return ref + "_q" + str(length) + "bp.fna"

def mummer_query_string(ref, length):
	return ref + "_q" + str(length) + "bp.fna"

def runCmatch(ref, query, matchlen, onCPU, outfile):

	if onCPU:
		cpuFlag = "-C"
		statfile = "%s-%ld.cpustats" % ( query, matchlen )
	else:
		cpuFlag = ""
		statfile = "%s-%ld.gpustats" % ( query, matchlen )

	
	match_flags = " -l %d " % (matchlen)
	
	cmd = "mummergpu %s -s %s -M -b %s  %s %s > %s" % (cpuFlag,
													statfile,
													match_flags,
													ref,
													query,
													outfile)
 	print >> sys.stderr, cmd
 	t = timeit.Timer("os.system('" + cmd + "')", "import os")
 	app_times = t.repeat(1, 1)
 	app_time_avg = sum(app_times)/len(app_times)
 	stats = get_stats(statfile)
	return (app_time_avg, stats)

def runMUMmer(ref, query, matchlen, outfile):
	query_file = query
	#for i in range(0,5):
	#	query_files += query + " "
		
	cmd = "mummer -maxmatch -b -l %d %s %s 2>/dev/null  > %s" % (matchlen,
															  ref,
															  query_file,
															  outfile)
														
 	print >> sys.stderr , cmd
 	t = timeit.Timer("os.system('" + cmd + "')", "import os")
 	app_times = t.repeat(1, 1)
 	app_time_avg = sum(app_times)/len(app_times)
 
	return app_time_avg

def get_gpu_time(stats):
	return stats["Kernel"] + stats["Copy queries to GPU"] + stats["Copy output from GPU"] + stats["Copy suffix tree to GPU"]

def execute_synth_trial(fout, directory, ref, querylen, matchlen):
	#cmatch_query_file = directory + "/" + ref + query

	query_file = "%s/%s_q%dbp.fna" % (directory, ref, querylen)
	ref_file = "%s/%s.fna" % (directory, ref)
	cpu_outfile = query_file + "-cpu.out"
	gpu_outfile = query_file + "-gpu.out"
	mummer_outfile = query_file + "-mummer.out"
	
	(gpu_app_time_avg, gpu_stats) = runCmatch(ref_file,
											  query_file,
											  matchlen,
											  False,
											  gpu_outfile)
	
	(cpu_app_time_avg, cpu_stats) = runCmatch(ref_file,
											  query_file,
											  matchlen,
											  True,
											  cpu_outfile)

	mummer_app_time_avg = runMUMmer(ref_file, query_file, matchlen == 0 and querylen or matchlen, mummer_outfile)
	

 	kernel_speedup = cpu_stats["Kernel"] / get_gpu_time(gpu_stats)

	print >>fout, querylen, matchlen, cpu_app_time_avg,gpu_app_time_avg, mummer_app_time_avg, cpu_app_time_avg / gpu_app_time_avg, mummer_app_time_avg / gpu_app_time_avg, kernel_speedup
	#print cmd



def execute_real_trial(fout, directory, ref, query, matchlen):
	
	ref_file = "%s/%s.fna" % (directory, ref)
	query_file = "%s/%s.fna" % (directory, query)
	cpu_outfile = query_file + "-cpu.out"
	gpu_outfile = query_file + "-gpu.out"
	mummer_outfile = query_file + "-mummer.out"
	
	(gpu_app_time_avg, gpu_stats) = runCmatch(ref_file,
											  query_file,
											  matchlen,
											  False,
											  gpu_outfile)
	
 	(cpu_app_time_avg, cpu_stats) = runCmatch(ref_file,
 											  query_file,
 											  matchlen,
 											  True,
											  cpu_outfile)

	mummer_app_time_avg = runMUMmer(ref_file, query_file, matchlen, mummer_outfile)
	


 	kernel_speedup = cpu_stats["Kernel"] /  get_gpu_time(gpu_stats)

   	print >> fout, "-", matchlen,cpu_app_time_avg,gpu_app_time_avg, mummer_app_time_avg, cpu_app_time_avg / gpu_app_time_avg, mummer_app_time_avg / gpu_app_time_avg, kernel_speedup


def run_different_query_lengths():
	f = open("anthrax/speedup.out", "w")
 	print >> f, "QUERY,MATCH_LENGTH,CPU,GPU,MUMMER,CPU_SPEEDUP,MUMMER_SPEEDUP,KERNEL_SPEEDUP"	
 	execute_synth_trial(f,"anthrax", "NC_003997", 25, 25)
 	execute_synth_trial(f,"anthrax", "NC_003997", 50, 50)
 	execute_synth_trial(f,"anthrax", "NC_003997", 100,100)
 	execute_synth_trial(f,"anthrax", "NC_003997", 200,200) 
 	execute_synth_trial(f,"anthrax", "NC_003997", 400,400)
 	execute_synth_trial(f,"anthrax", "NC_003997", 800,800)
	
def run_ssuis_solexa():
	f = open("s_suis/speedup.out", "w")
 	print >> f, "QUERY,MATCH_LENGTH,CPU,GPU,MUMMER,CPU_SPEEDUP,MUMMER_SPEEDUP,KERNEL_SPEEDUP"	
	execute_real_trial(f,"s_suis", "cleanref", "cleanreads", 20)
	
def run_cereus_454():
	f = open("cereus/speedup.out", "w")
 	print >> f, "QUERY,MATCH_LENGTH,CPU,GPU,MUMMER,CPU_SPEEDUP,MUMMER_SPEEDUP,KERNEL_SPEEDUP"	
	execute_real_trial(f,"cereus", "cleanref", "cleanreads", 20)

def run_briggsae_sanger():
	f = open("cbriggsae/speedup.out", "w")
 	print >> f, "QUERY,MATCH_LENGTH,CPU,GPU,MUMMER,CPU_SPEEDUP,MUMMER_SPEEDUP,KERNEL_SPEEDUP"	
	execute_real_trial(f,"cbriggsae", "cleanref", "cleanreads", 100)

def main(argv=None):
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(argv[1:], "hv", ["help"])
		except getopt.error, msg:
			raise Usage(msg)
	
		for option, value in opts:
			if option == "-v":
				verbose = True
			if option in ("-h", "--help"):
				raise Usage(help_message)

		run_ssuis_solexa()
		run_cereus_454()
		run_briggsae_sanger()

		run_different_query_lengths()
		
	except Usage, err:
		print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
		print >> sys.stderr, "\t for help use --help"
		return 2


if __name__ == "__main__":
	sys.exit(main())
